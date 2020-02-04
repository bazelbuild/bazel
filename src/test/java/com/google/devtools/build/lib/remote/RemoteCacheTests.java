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
import static com.google.devtools.build.lib.remote.util.DigestUtil.toBinaryDigest;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Charsets;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.RemoteCache.OutputFilesLocker;
import com.google.devtools.build.lib.remote.RemoteCache.UploadManifest;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests for {@link RemoteCache}. */
@RunWith(JUnit4.class)
public class RemoteCacheTests {

  @Mock private OutputFilesLocker outputFilesLocker;

  private FileSystem fs;
  private Path execRoot;
  ArtifactRoot artifactRoot;
  private final DigestUtil digestUtil = new DigestUtil(DigestHashFunction.SHA256);
  private FakeActionInputFileCache fakeFileCache;

  private static ListeningScheduledExecutorService retryService;

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot");
    execRoot.createDirectoryAndParents();
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, execRoot.getChild("outputs"));
    artifactRoot.getRoot().asPath().createDirectoryAndParents();
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Test
  public void uploadAbsoluteFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("link").setDigest(digest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadAbsoluteDirectorySymlinkAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path foo = fs.getPath("/execroot/dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/link");
    link.createSymbolicLink(dir);

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, fs.getPath("/execroot/link/foo"));

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(FileNode.newBuilder().setName("foo").setDigest(digest)))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("link").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ false, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("link").setDigest(digest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeDirectorySymlinkAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path foo = fs.getPath("/execroot/dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/link");
    link.createSymbolicLink(dir.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ false, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, fs.getPath("/execroot/link/foo"));

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(FileNode.newBuilder().setName("foo").setDigest(digest)))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("link").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeFileSymlink() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFileSymlinksBuilder().setPath("link").setTarget("target");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeDirectorySymlink() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path file = fs.getPath("/execroot/dir/foo");
    FileSystemUtils.writeContent(file, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/link");
    link.createSymbolicLink(dir.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectorySymlinksBuilder().setPath("link").setTarget("dir");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadDanglingSymlinkError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/link");
    assertThat(e).hasMessageThat().contains("target");
  }

  @Test
  public void uploadSymlinksNoAllowError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ false);
    ExecException e = assertThrows(ExecException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("symbolic link");
    assertThat(e).hasMessageThat().contains("--remote_allow_symlink_upload");
  }

  @Test
  public void uploadAbsoluteFileSymlinkInDirectoryAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(FileNode.newBuilder().setName("link").setDigest(digest)))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadAbsoluteDirectorySymlinkInDirectoryAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path bardir = fs.getPath("/execroot/bardir");
    bardir.createDirectory();
    Path foo = fs.getPath("/execroot/bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(bardir);

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, fs.getPath("/execroot/dir/link/foo"));

    Directory barDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("foo").setDigest(digest))
            .build();
    Digest barDigest = digestUtil.compute(barDir);
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addDirectories(
                        DirectoryNode.newBuilder().setName("link").setDigest(barDigest)))
            .addChildren(barDir)
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeFileSymlinkInDirectoryAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(PathFragment.create("../target"));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ false, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(FileNode.newBuilder().setName("link").setDigest(digest)))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeDirectorySymlinkInDirectoryAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path bardir = fs.getPath("/execroot/bardir");
    bardir.createDirectory();
    Path foo = fs.getPath("/execroot/bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ false, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, fs.getPath("/execroot/dir/link/foo"));

    Directory barDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("foo").setDigest(digest))
            .build();
    Digest barDigest = digestUtil.compute(barDir);
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addDirectories(
                        DirectoryNode.newBuilder().setName("link").setDigest(barDigest)))
            .addChildren(barDir)
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeFileSymlinkInDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(PathFragment.create("../target"));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    assertThat(um.getDigestToFile()).isEmpty();

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("../target")))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeDirectorySymlinkInDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path bardir = fs.getPath("/execroot/bardir");
    bardir.createDirectory();
    Path foo = fs.getPath("/execroot/bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    assertThat(um.getDigestToFile()).isEmpty();

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("../bardir")))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadDanglingSymlinkInDirectoryError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ true);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/dir/link");
    assertThat(e).hasMessageThat().contains("/execroot/target");
  }

  @Test
  public void uploadSymlinkInDirectoryNoAllowError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil, result, execRoot, /*uploadSymlinks=*/ true, /*allowSymlinks=*/ false);
    ExecException e = assertThrows(ExecException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("symbolic link");
    assertThat(e).hasMessageThat().contains("dir/link");
    assertThat(e).hasMessageThat().contains("--remote_allow_symlink_upload");
  }

  @Test
  public void downloadRelativeFileSymlink() throws Exception {
    RemoteCache cache = newRemoteCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFileSymlinksBuilder().setPath("a/b/link").setTarget("../../foo");
    // Doesn't check for dangling links, hence download succeeds.
    cache.download(result.build(), execRoot, null, outputFilesLocker);
    Path path = execRoot.getRelative("a/b/link");
    assertThat(path.isSymbolicLink()).isTrue();
    assertThat(path.readSymbolicLink()).isEqualTo(PathFragment.create("../../foo"));
    verify(outputFilesLocker).lock();
  }

  @Test
  public void downloadRelativeDirectorySymlink() throws Exception {
    RemoteCache cache = newRemoteCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectorySymlinksBuilder().setPath("a/b/link").setTarget("foo");
    // Doesn't check for dangling links, hence download succeeds.
    cache.download(result.build(), execRoot, null, outputFilesLocker);
    Path path = execRoot.getRelative("a/b/link");
    assertThat(path.isSymbolicLink()).isTrue();
    assertThat(path.readSymbolicLink()).isEqualTo(PathFragment.create("foo"));
    verify(outputFilesLocker).lock();
  }

  @Test
  public void downloadRelativeSymlinkInDirectory() throws Exception {
    InMemoryRemoteCache cache = newRemoteCache();
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("../foo")))
            .build();
    Digest treeDigest = cache.addContents(tree.toByteArray());
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    // Doesn't check for dangling links, hence download succeeds.
    cache.download(result.build(), execRoot, null, outputFilesLocker);
    Path path = execRoot.getRelative("dir/link");
    assertThat(path.isSymbolicLink()).isTrue();
    assertThat(path.readSymbolicLink()).isEqualTo(PathFragment.create("../foo"));
    verify(outputFilesLocker).lock();
  }

  @Test
  public void downloadAbsoluteDirectorySymlinkError() throws Exception {
    RemoteCache cache = newRemoteCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectorySymlinksBuilder().setPath("foo").setTarget("/abs/link");
    IOException expected =
        assertThrows(
            IOException.class,
            () -> cache.download(result.build(), execRoot, null, outputFilesLocker));
    assertThat(expected).hasMessageThat().contains("/abs/link");
    assertThat(expected).hasMessageThat().contains("absolute path");
    verify(outputFilesLocker).lock();
  }

  @Test
  public void downloadAbsoluteFileSymlinkError() throws Exception {
    RemoteCache cache = newRemoteCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFileSymlinksBuilder().setPath("foo").setTarget("/abs/link");
    IOException expected =
        assertThrows(
            IOException.class,
            () -> cache.download(result.build(), execRoot, null, outputFilesLocker));
    assertThat(expected).hasMessageThat().contains("/abs/link");
    assertThat(expected).hasMessageThat().contains("absolute path");
    verify(outputFilesLocker).lock();
  }

  @Test
  public void downloadAbsoluteSymlinkInDirectoryError() throws Exception {
    InMemoryRemoteCache cache = newRemoteCache();
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("/foo")))
            .build();
    Digest treeDigest = cache.addContents(tree.toByteArray());
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    IOException expected =
        assertThrows(
            IOException.class,
            () -> cache.download(result.build(), execRoot, null, outputFilesLocker));
    assertThat(expected.getSuppressed()).isEmpty();
    assertThat(expected).hasMessageThat().contains("dir/link");
    assertThat(expected).hasMessageThat().contains("/foo");
    assertThat(expected).hasMessageThat().contains("absolute path");
    verify(outputFilesLocker).lock();
  }

  @Test
  public void downloadFailureMaintainsDirectories() throws Exception {
    InMemoryRemoteCache cache = newRemoteCache();
    Tree tree = Tree.newBuilder().setRoot(Directory.newBuilder()).build();
    Digest treeDigest = cache.addContents(tree.toByteArray());
    Digest outputFileDigest =
        cache.addException("outputdir/outputfile", new IOException("download failed"));
    Digest otherFileDigest = cache.addContents("otherfile");

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("outputdir").setTreeDigest(treeDigest);
    result.addOutputFiles(
        OutputFile.newBuilder().setPath("outputdir/outputfile").setDigest(outputFileDigest));
    result.addOutputFiles(OutputFile.newBuilder().setPath("otherfile").setDigest(otherFileDigest));
    assertThrows(
        IOException.class, () -> cache.download(result.build(), execRoot, null, outputFilesLocker));
    assertThat(cache.getNumFailedDownloads()).isEqualTo(1);
    assertThat(execRoot.getRelative("outputdir").exists()).isTrue();
    assertThat(execRoot.getRelative("outputdir/outputfile").exists()).isFalse();
    assertThat(execRoot.getRelative("otherfile").exists()).isFalse();
    verify(outputFilesLocker, never()).lock();
  }

  @Test
  public void onErrorWaitForRemainingDownloadsToComplete() throws Exception {
    // If one or more downloads of output files / directories fail then the code should
    // wait for all downloads to have been completed before it tries to clean up partially
    // downloaded files.

    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");

    InMemoryRemoteCache cache = newRemoteCache();
    Digest digest1 = cache.addContents("file1");
    Digest digest2 = cache.addException("file2", new IOException("download failed"));
    Digest digest3 = cache.addContents("file3");

    ActionResult result =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("file1").setDigest(digest1))
            .addOutputFiles(OutputFile.newBuilder().setPath("file2").setDigest(digest2))
            .addOutputFiles(OutputFile.newBuilder().setPath("file3").setDigest(digest3))
            .build();
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                cache.download(
                    result, execRoot, new FileOutErr(stdout, stderr), outputFilesLocker));
    assertThat(e.getSuppressed()).isEmpty();
    assertThat(cache.getNumSuccessfulDownloads()).isEqualTo(2);
    assertThat(cache.getNumFailedDownloads()).isEqualTo(1);
    assertThat(Throwables.getRootCause(e)).hasMessageThat().isEqualTo("download failed");
    verify(outputFilesLocker, never()).lock();
  }

  @Test
  public void downloadWithMultipleErrorsAddsThemAsSuppressed() throws Exception {
    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");

    InMemoryRemoteCache cache = newRemoteCache();
    Digest digest1 = cache.addContents("file1");
    Digest digest2 = cache.addException("file2", new IOException("file2 failed"));
    Digest digest3 = cache.addException("file3", new IOException("file3 failed"));

    ActionResult result =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("file1").setDigest(digest1))
            .addOutputFiles(OutputFile.newBuilder().setPath("file2").setDigest(digest2))
            .addOutputFiles(OutputFile.newBuilder().setPath("file3").setDigest(digest3))
            .build();
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                cache.download(
                    result, execRoot, new FileOutErr(stdout, stderr), outputFilesLocker));

    assertThat(e.getSuppressed()).hasLength(1);
    assertThat(e.getSuppressed()[0]).isInstanceOf(IOException.class);
    assertThat(e.getSuppressed()[0]).hasMessageThat().isEqualTo("file3 failed");
    assertThat(Throwables.getRootCause(e)).hasMessageThat().isEqualTo("file2 failed");
  }

  @Test
  public void downloadWithDuplicateIOErrorsDoesNotSuppress() throws Exception {
    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");

    InMemoryRemoteCache cache = newRemoteCache();
    Digest digest1 = cache.addContents("file1");
    IOException reusedException = new IOException("reused io exception");
    Digest digest2 = cache.addException("file2", reusedException);
    Digest digest3 = cache.addException("file3", reusedException);

    ActionResult result =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("file1").setDigest(digest1))
            .addOutputFiles(OutputFile.newBuilder().setPath("file2").setDigest(digest2))
            .addOutputFiles(OutputFile.newBuilder().setPath("file3").setDigest(digest3))
            .build();
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                cache.download(
                    result, execRoot, new FileOutErr(stdout, stderr), outputFilesLocker));

    assertThat(e.getSuppressed()).isEmpty();
    assertThat(Throwables.getRootCause(e)).hasMessageThat().isEqualTo("reused io exception");
  }

  @Test
  public void downloadWithDuplicateInterruptionsDoesNotSuppress() throws Exception {
    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");

    InMemoryRemoteCache cache = newRemoteCache();
    Digest digest1 = cache.addContents("file1");
    InterruptedException reusedInterruption = new InterruptedException("reused interruption");
    Digest digest2 = cache.addException("file2", reusedInterruption);
    Digest digest3 = cache.addException("file3", reusedInterruption);

    ActionResult result =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("file1").setDigest(digest1))
            .addOutputFiles(OutputFile.newBuilder().setPath("file2").setDigest(digest2))
            .addOutputFiles(OutputFile.newBuilder().setPath("file3").setDigest(digest3))
            .build();
    InterruptedException e =
        assertThrows(
            InterruptedException.class,
            () ->
                cache.download(
                    result, execRoot, new FileOutErr(stdout, stderr), outputFilesLocker));

    assertThat(e.getSuppressed()).isEmpty();
    assertThat(Throwables.getRootCause(e)).hasMessageThat().isEqualTo("reused interruption");
  }

  @Test
  public void testDownloadWithStdoutStderrOnSuccess() throws Exception {
    // Tests that fetching stdout/stderr as a digest works and that OutErr is still
    // writable afterwards.
    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");
    FileOutErr outErr = new FileOutErr(stdout, stderr);
    FileOutErr childOutErr = outErr.childOutErr();
    FileOutErr spyOutErr = Mockito.spy(outErr);
    FileOutErr spyChildOutErr = Mockito.spy(childOutErr);
    when(spyOutErr.childOutErr()).thenReturn(spyChildOutErr);

    InMemoryRemoteCache cache = newRemoteCache();
    Digest digestStdout = cache.addContents("stdout");
    Digest digestStderr = cache.addContents("stderr");

    ActionResult result =
        ActionResult.newBuilder()
            .setExitCode(0)
            .setStdoutDigest(digestStdout)
            .setStderrDigest(digestStderr)
            .build();

    cache.download(result, execRoot, spyOutErr, outputFilesLocker);

    verify(spyOutErr, Mockito.times(2)).childOutErr();
    verify(spyChildOutErr).clearOut();
    verify(spyChildOutErr).clearErr();
    assertThat(outErr.getOutputPath().exists()).isTrue();
    assertThat(outErr.getErrorPath().exists()).isTrue();

    try {
      outErr.getOutputStream().write(0);
      outErr.getErrorStream().write(0);
    } catch (IOException err) {
      throw new AssertionError("outErr should still be writable after download finished.", err);
    }

    verify(outputFilesLocker).lock();
  }

  @Test
  public void testDownloadWithStdoutStderrOnFailure() throws Exception {
    // Test that when downloading stdout/stderr fails the OutErr is still writable
    // and empty.
    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");
    FileOutErr outErr = new FileOutErr(stdout, stderr);
    FileOutErr childOutErr = outErr.childOutErr();
    FileOutErr spyOutErr = Mockito.spy(outErr);
    FileOutErr spyChildOutErr = Mockito.spy(childOutErr);
    when(spyOutErr.childOutErr()).thenReturn(spyChildOutErr);

    InMemoryRemoteCache cache = newRemoteCache();
    // Don't add stdout/stderr as a known blob to the remote cache so that downloading it will fail
    Digest digestStdout = digestUtil.computeAsUtf8("stdout");
    Digest digestStderr = digestUtil.computeAsUtf8("stderr");

    ActionResult result =
        ActionResult.newBuilder()
            .setExitCode(0)
            .setStdoutDigest(digestStdout)
            .setStderrDigest(digestStderr)
            .build();
    assertThrows(
        IOException.class, () -> cache.download(result, execRoot, spyOutErr, outputFilesLocker));
    verify(spyOutErr, Mockito.times(2)).childOutErr();
    verify(spyChildOutErr).clearOut();
    verify(spyChildOutErr).clearErr();
    assertThat(outErr.getOutputPath().exists()).isFalse();
    assertThat(outErr.getErrorPath().exists()).isFalse();

    try {
      outErr.getOutputStream().write(0);
      outErr.getErrorStream().write(0);
    } catch (IOException err) {
      throw new AssertionError("outErr should still be writable after download failed.", err);
    }

    verify(outputFilesLocker, never()).lock();
  }

  @Test
  public void testDownloadClashes() throws Exception {
    // Test that injecting the metadata for a remote output file works

    // arrange
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/foo.tmp").setDigest(d1))
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/foo").setDigest(d2))
            .build();

    Artifact a1 = ActionsTestUtil.createArtifact(artifactRoot, "foo.tmp");
    Artifact a2 = ActionsTestUtil.createArtifact(artifactRoot, "foo");

    // act

    remoteCache.download(r, execRoot, new FileOutErr(), outputFilesLocker);

    // assert

    assertThat(FileSystemUtils.readContent(a1.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("content1");
    assertThat(FileSystemUtils.readContent(a2.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("content2");
    verify(outputFilesLocker).lock();
  }

  @Test
  public void testDownloadMinimalFiles() throws Exception {
    // Test that injecting the metadata for a remote output file works

    // arrange
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file1").setDigest(d1))
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file2").setDigest(d2))
            .build();

    Artifact a1 = ActionsTestUtil.createArtifact(artifactRoot, "file1");
    Artifact a2 = ActionsTestUtil.createArtifact(artifactRoot, "file2");

    MetadataInjector injector = mock(MetadataInjector.class);

    // act
    InMemoryOutput inMemoryOutput =
        remoteCache.downloadMinimal(
            r,
            ImmutableList.of(a1, a2),
            /* inMemoryOutputPath= */ null,
            new FileOutErr(),
            execRoot,
            injector,
            outputFilesLocker);

    // assert
    assertThat(inMemoryOutput).isNull();
    verify(injector)
        .injectRemoteFile(eq(a1), eq(toBinaryDigest(d1)), eq(d1.getSizeBytes()), anyInt());
    verify(injector)
        .injectRemoteFile(eq(a2), eq(toBinaryDigest(d2)), eq(d2.getSizeBytes()), anyInt());

    Path outputBase = artifactRoot.getRoot().asPath();
    assertThat(outputBase.readdir(Symlinks.NOFOLLOW)).isEmpty();

    verify(outputFilesLocker).lock();
  }

  @Test
  public void testDownloadMinimalDirectory() throws Exception {
    // Test that injecting the metadata for a tree artifact / remote output directory works

    // arrange
    InMemoryRemoteCache remoteCache = newRemoteCache();
    // Output Directory:
    // dir/file1
    // dir/a/file2
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    FileNode file1 = FileNode.newBuilder().setName("file1").setDigest(d1).build();
    FileNode file2 = FileNode.newBuilder().setName("file2").setDigest(d2).build();
    Directory a = Directory.newBuilder().addFiles(file2).build();
    Digest da = remoteCache.addContents(a);
    Directory root =
        Directory.newBuilder()
            .addFiles(file1)
            .addDirectories(DirectoryNode.newBuilder().setName("a").setDigest(da))
            .build();
    Tree t = Tree.newBuilder().setRoot(root).addChildren(a).build();
    Digest dt = remoteCache.addContents(t);
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputDirectories(
                OutputDirectory.newBuilder().setPath("outputs/dir").setTreeDigest(dt))
            .build();

    SpecialArtifact dir =
        new SpecialArtifact(
            artifactRoot,
            PathFragment.create("outputs/dir"),
            ActionsTestUtil.NULL_ARTIFACT_OWNER,
            SpecialArtifactType.TREE);

    MetadataInjector injector = mock(MetadataInjector.class);

    // act
    InMemoryOutput inMemoryOutput =
        remoteCache.downloadMinimal(
            r,
            ImmutableList.of(dir),
            /* inMemoryOutputPath= */ null,
            new FileOutErr(),
            execRoot,
            injector,
            outputFilesLocker);

    // assert
    assertThat(inMemoryOutput).isNull();

    Map<PathFragment, RemoteFileArtifactValue> m =
        ImmutableMap.<PathFragment, RemoteFileArtifactValue>builder()
            .put(
                PathFragment.create("file1"),
                new RemoteFileArtifactValue(toBinaryDigest(d1), d1.getSizeBytes(), 1))
            .put(
                PathFragment.create("a/file2"),
                new RemoteFileArtifactValue(toBinaryDigest(d2), d2.getSizeBytes(), 1))
            .build();
    verify(injector).injectRemoteDirectory(eq(dir), eq(m));

    Path outputBase = artifactRoot.getRoot().asPath();
    assertThat(outputBase.readdir(Symlinks.NOFOLLOW)).isEmpty();

    verify(outputFilesLocker).lock();
  }

  @Test
  public void testDownloadMinimalDirectoryFails() throws Exception {
    // Test that we properly fail when downloading the metadata of an output
    // directory fails

    // arrange
    InMemoryRemoteCache remoteCache = newRemoteCache();
    // Output Directory:
    // dir/file1
    // dir/a/file2
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    FileNode file1 = FileNode.newBuilder().setName("file1").setDigest(d1).build();
    FileNode file2 = FileNode.newBuilder().setName("file2").setDigest(d2).build();
    Directory a = Directory.newBuilder().addFiles(file2).build();
    Digest da = remoteCache.addContents(a);
    Directory root =
        Directory.newBuilder()
            .addFiles(file1)
            .addDirectories(DirectoryNode.newBuilder().setName("a").setDigest(da))
            .build();
    Tree t = Tree.newBuilder().setRoot(root).addChildren(a).build();
    // Downloading the tree will fail
    IOException downloadTreeException = new IOException("entry not found");
    Digest dt = remoteCache.addException(t, downloadTreeException);
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputDirectories(
                OutputDirectory.newBuilder().setPath("outputs/dir").setTreeDigest(dt))
            .build();
    SpecialArtifact dir =
        new SpecialArtifact(
            artifactRoot,
            PathFragment.create("outputs/dir"),
            ActionsTestUtil.NULL_ARTIFACT_OWNER,
            SpecialArtifactType.TREE);
    MetadataInjector injector = mock(MetadataInjector.class);

    // act
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                remoteCache.downloadMinimal(
                    r,
                    ImmutableList.of(dir),
                    /* inMemoryOutputPath= */ null,
                    new FileOutErr(),
                    execRoot,
                    injector,
                    outputFilesLocker));
    assertThat(e).isEqualTo(downloadTreeException);

    verify(outputFilesLocker, never()).lock();
  }

  @Test
  public void testDownloadMinimalWithStdoutStderr() throws Exception {
    // Test that downloading of non-embedded stdout and stderr works

    // arrange
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest dOut = remoteCache.addContents("stdout");
    Digest dErr = remoteCache.addContents("stderr");
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .setStdoutDigest(dOut)
            .setStderrDigest(dErr)
            .build();
    RecordingOutErr outErr = new RecordingOutErr();
    MetadataInjector injector = mock(MetadataInjector.class);

    // act
    InMemoryOutput inMemoryOutput =
        remoteCache.downloadMinimal(
            r,
            ImmutableList.of(),
            /* inMemoryOutputPath= */ null,
            outErr,
            execRoot,
            injector,
            outputFilesLocker);

    // assert
    assertThat(inMemoryOutput).isNull();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");

    Path outputBase = artifactRoot.getRoot().asPath();
    assertThat(outputBase.readdir(Symlinks.NOFOLLOW)).isEmpty();

    verify(outputFilesLocker).lock();
  }

  @Test
  public void testDownloadMinimalWithInMemoryOutput() throws Exception {
    // Test that downloading an in memory output works

    // arrange
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file1").setDigest(d1))
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file2").setDigest(d2))
            .build();
    Artifact a1 = ActionsTestUtil.createArtifact(artifactRoot, "file1");
    Artifact a2 = ActionsTestUtil.createArtifact(artifactRoot, "file2");
    MetadataInjector injector = mock(MetadataInjector.class);
    // a1 should be provided as an InMemoryOutput
    PathFragment inMemoryOutputPathFragment = a1.getPath().relativeTo(execRoot);

    // act
    InMemoryOutput inMemoryOutput =
        remoteCache.downloadMinimal(
            r,
            ImmutableList.of(a1, a2),
            inMemoryOutputPathFragment,
            new FileOutErr(),
            execRoot,
            injector,
            outputFilesLocker);

    // assert
    assertThat(inMemoryOutput).isNotNull();
    ByteString expectedContents = ByteString.copyFrom("content1", UTF_8);
    assertThat(inMemoryOutput.getContents()).isEqualTo(expectedContents);
    assertThat(inMemoryOutput.getOutput()).isEqualTo(a1);
    // The in memory file also needs to be injected as an output
    verify(injector)
        .injectRemoteFile(eq(a1), eq(toBinaryDigest(d1)), eq(d1.getSizeBytes()), anyInt());
    verify(injector)
        .injectRemoteFile(eq(a2), eq(toBinaryDigest(d2)), eq(d2.getSizeBytes()), anyInt());

    Path outputBase = artifactRoot.getRoot().asPath();
    assertThat(outputBase.readdir(Symlinks.NOFOLLOW)).isEmpty();

    verify(outputFilesLocker).lock();
  }

  @Test
  public void testDownloadEmptyBlobAndFile() throws Exception {
    // Test that downloading an empty BLOB/file does not try to perform a download.

    // arrange
    Path file = fs.getPath("/execroot/file");
    RemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);

    // act and assert
    assertThat(Utils.getFromFuture(remoteCache.downloadBlob(emptyDigest))).isEmpty();

    try (OutputStream out = file.getOutputStream()) {
      Utils.getFromFuture(remoteCache.downloadFile(file, emptyDigest));
    }
    assertThat(file.exists()).isTrue();
    assertThat(file.getFileSize()).isEqualTo(0);
  }

  @Test
  public void testDownloadDirectory() throws Exception {
    // Test that downloading an output directory works.

    // arrange
    final ConcurrentMap<Digest, byte[]> cas = new ConcurrentHashMap<>();

    Digest fooDigest = digestUtil.computeAsUtf8("foo-contents");
    cas.put(fooDigest, "foo-contents".getBytes(Charsets.UTF_8));
    Digest quxDigest = digestUtil.computeAsUtf8("qux-contents");
    cas.put(quxDigest, "qux-contents".getBytes(Charsets.UTF_8));

    Tree barTreeMessage =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder()
                            .setName("qux")
                            .setDigest(quxDigest)
                            .setIsExecutable(true)))
            .build();
    Digest barTreeDigest = digestUtil.compute(barTreeMessage);
    cas.put(barTreeDigest, barTreeMessage.toByteArray());

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);

    // act
    RemoteCache remoteCache = newRemoteCache(cas);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    // assert
    assertThat(digestUtil.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(digestUtil.compute(execRoot.getRelative("a/bar/qux"))).isEqualTo(quxDigest);
    assertThat(execRoot.getRelative("a/bar/qux").isExecutable()).isTrue();
  }

  @Test
  public void testDownloadEmptyDirectory() throws Exception {
    // Test that downloading an empty output directory works.

    // arrange
    Tree barTreeMessage = Tree.newBuilder().setRoot(Directory.newBuilder()).build();
    Digest barTreeDigest = digestUtil.compute(barTreeMessage);

    final ConcurrentMap<Digest, byte[]> map = new ConcurrentHashMap<>();
    map.put(barTreeDigest, barTreeMessage.toByteArray());

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);

    // act
    RemoteCache remoteCache = newRemoteCache(map);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    // assert
    assertThat(execRoot.getRelative("a/bar").isDirectory()).isTrue();
  }

  @Test
  public void testDownloadNestedDirectory() throws Exception {
    // Test that downloading a nested output directory works.

    // arrange
    Digest fooDigest = digestUtil.computeAsUtf8("foo-contents");
    Digest quxDigest = digestUtil.computeAsUtf8("qux-contents");
    Directory wobbleDirMessage =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("qux").setDigest(quxDigest))
            .build();
    Digest wobbleDirDigest = digestUtil.compute(wobbleDirMessage);
    Tree barTreeMessage =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder()
                            .setName("qux")
                            .setDigest(quxDigest)
                            .setIsExecutable(true))
                    .addDirectories(
                        DirectoryNode.newBuilder().setName("wobble").setDigest(wobbleDirDigest)))
            .addChildren(wobbleDirMessage)
            .build();
    Digest barTreeDigest = digestUtil.compute(barTreeMessage);

    final ConcurrentMap<Digest, byte[]> map = new ConcurrentHashMap<>();
    map.put(fooDigest, "foo-contents".getBytes(Charsets.UTF_8));
    map.put(barTreeDigest, barTreeMessage.toByteArray());
    map.put(quxDigest, "qux-contents".getBytes(Charsets.UTF_8));

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    result.addOutputDirectoriesBuilder().setPath("a/bar").setTreeDigest(barTreeDigest);

    // act
    RemoteCache remoteCache = newRemoteCache(map);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    // assert
    assertThat(digestUtil.compute(execRoot.getRelative("a/foo"))).isEqualTo(fooDigest);
    assertThat(digestUtil.compute(execRoot.getRelative("a/bar/wobble/qux"))).isEqualTo(quxDigest);
    assertThat(execRoot.getRelative("a/bar/wobble/qux").isExecutable()).isFalse();
  }

  @Test
  public void testDownloadDirectoryWithSameHash() throws Exception {
    // Test that downloading an output directory works when two Directory
    // protos have the same hash i.e. because they have the same name and contents or are empty.

    /*
     * /bar/foo/file
     * /foo/file
     */

    // arrange
    Digest fileDigest = digestUtil.computeAsUtf8("file");
    FileNode file = FileNode.newBuilder().setName("file").setDigest(fileDigest).build();
    Directory fooDir = Directory.newBuilder().addFiles(file).build();
    Digest fooDigest = digestUtil.compute(fooDir);
    DirectoryNode fooDirNode =
        DirectoryNode.newBuilder().setName("foo").setDigest(fooDigest).build();
    Directory barDir = Directory.newBuilder().addDirectories(fooDirNode).build();
    Digest barDigest = digestUtil.compute(barDir);
    DirectoryNode barDirNode =
        DirectoryNode.newBuilder().setName("bar").setDigest(barDigest).build();
    Directory rootDir =
        Directory.newBuilder().addDirectories(fooDirNode).addDirectories(barDirNode).build();

    Tree tree =
        Tree.newBuilder()
            .setRoot(rootDir)
            .addChildren(barDir)
            .addChildren(fooDir)
            .addChildren(fooDir)
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    final ConcurrentMap<Digest, byte[]> map = new ConcurrentHashMap<>();
    map.put(fileDigest, "file".getBytes(Charsets.UTF_8));
    map.put(treeDigest, tree.toByteArray());

    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("a/").setTreeDigest(treeDigest);

    // act
    RemoteCache remoteCache = newRemoteCache(map);
    remoteCache.download(result.build(), execRoot, null, /* outputFilesLocker= */ () -> {});

    // assert
    assertThat(digestUtil.compute(execRoot.getRelative("a/bar/foo/file"))).isEqualTo(fileDigest);
    assertThat(digestUtil.compute(execRoot.getRelative("a/foo/file"))).isEqualTo(fileDigest);
  }

  @Test
  public void testUploadDirectory() throws Exception {
    // Test that uploading a directory works.

    // arrange
    Digest fooDigest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("a/foo"), "xyz");
    Digest quxDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/qux"), "abc");
    Digest barDigest =
        fakeFileCache.createScratchInputDirectory(
            ActionInputHelper.fromPath("bar"),
            Tree.newBuilder()
                .setRoot(
                    Directory.newBuilder()
                        .addFiles(
                            FileNode.newBuilder()
                                .setIsExecutable(true)
                                .setName("qux")
                                .setDigest(quxDigest)
                                .build())
                        .build())
                .build());
    Path fooFile = execRoot.getRelative("a/foo");
    Path quxFile = execRoot.getRelative("bar/qux");
    quxFile.setExecutable(true);
    Path barDir = execRoot.getRelative("bar");
    Command cmd = Command.newBuilder().addOutputFiles("bla").build();
    Digest cmdDigest = digestUtil.compute(cmd);
    Action action = Action.newBuilder().setCommandDigest(cmdDigest).build();
    Digest actionDigest = digestUtil.compute(action);

    // act
    InMemoryRemoteCache remoteCache = newRemoteCache();
    ActionResult result =
        remoteCache.upload(
            digestUtil.asActionKey(actionDigest),
            action,
            cmd,
            execRoot,
            ImmutableList.of(fooFile, barDir),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());

    ImmutableList<Digest> toQuery =
        ImmutableList.of(fooDigest, quxDigest, barDigest, cmdDigest, actionDigest);
    assertThat(remoteCache.findMissingDigests(toQuery)).isEmpty();
  }

  @Test
  public void testUploadEmptyDirectory() throws Exception {
    // Test that uploading an empty directory works.

    // arrange
    final Digest barDigest =
        fakeFileCache.createScratchInputDirectory(
            ActionInputHelper.fromPath("bar"),
            Tree.newBuilder().setRoot(Directory.newBuilder().build()).build());
    final Path barDir = execRoot.getRelative("bar");
    Action action = Action.getDefaultInstance();
    ActionKey actionDigest = digestUtil.computeActionKey(action);
    Command cmd = Command.getDefaultInstance();

    // act
    InMemoryRemoteCache remoteCache = newRemoteCache();
    ActionResult result =
        remoteCache.upload(
            actionDigest,
            action,
            cmd,
            execRoot,
            ImmutableList.of(barDir),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());
    assertThat(remoteCache.findMissingDigests(ImmutableList.of(barDigest))).isEmpty();
  }

  @Test
  public void testUploadNestedDirectory() throws Exception {
    // Test that uploading a nested directory works.

    // arrange
    final Digest wobbleDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/test/wobble"), "xyz");
    final Digest quxDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/qux"), "abc");
    final Directory testDirMessage =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("wobble").setDigest(wobbleDigest).build())
            .build();
    final Digest testDigest = digestUtil.compute(testDirMessage);
    final Tree barTree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(
                        FileNode.newBuilder()
                            .setIsExecutable(true)
                            .setName("qux")
                            .setDigest(quxDigest))
                    .addDirectories(
                        DirectoryNode.newBuilder().setName("test").setDigest(testDigest)))
            .addChildren(testDirMessage)
            .build();
    final Digest barDigest =
        fakeFileCache.createScratchInputDirectory(ActionInputHelper.fromPath("bar"), barTree);

    final Path quxFile = execRoot.getRelative("bar/qux");
    quxFile.setExecutable(true);
    final Path barDir = execRoot.getRelative("bar");

    Action action = Action.getDefaultInstance();
    ActionKey actionDigest = digestUtil.computeActionKey(action);
    Command cmd = Command.getDefaultInstance();

    // act
    InMemoryRemoteCache remoteCache = newRemoteCache();
    ActionResult result =
        remoteCache.upload(
            actionDigest,
            action,
            cmd,
            execRoot,
            ImmutableList.of(barDir),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());

    ImmutableList<Digest> toQuery = ImmutableList.of(wobbleDigest, quxDigest, barDigest);
    assertThat(remoteCache.findMissingDigests(toQuery)).isEmpty();
  }

  private InMemoryRemoteCache newRemoteCache(Map<Digest, byte[]> casEntries) {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    return new InMemoryRemoteCache(casEntries, options, digestUtil);
  }

  private InMemoryRemoteCache newRemoteCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    return new InMemoryRemoteCache(options, digestUtil);
  }

  private static class InMemoryRemoteCache extends RemoteCache {

    InMemoryRemoteCache(
        Map<Digest, byte[]> casEntries, RemoteOptions options, DigestUtil digestUtil) {
      super(new InMemoryCacheClient(casEntries), options, digestUtil);
    }

    InMemoryRemoteCache(RemoteOptions options, DigestUtil digestUtil) {
      super(new InMemoryCacheClient(), options, digestUtil);
    }

    Digest addContents(String txt) throws IOException, InterruptedException {
      return addContents(txt.getBytes(UTF_8));
    }

    Digest addContents(byte[] bytes) throws IOException, InterruptedException {
      Digest digest = digestUtil.compute(bytes);
      Utils.getFromFuture(cacheProtocol.uploadBlob(digest, ByteString.copyFrom(bytes)));
      return digest;
    }

    Digest addContents(Message m) throws IOException, InterruptedException {
      return addContents(m.toByteArray());
    }

    Digest addException(String txt, Exception e) {
      Digest digest = digestUtil.compute(txt.getBytes(UTF_8));
      ((InMemoryCacheClient) cacheProtocol).addDownloadFailure(digest, e);
      return digest;
    }

    Digest addException(Message m, Exception e) {
      Digest digest = digestUtil.compute(m);
      ((InMemoryCacheClient) cacheProtocol).addDownloadFailure(digest, e);
      return digest;
    }

    int getNumSuccessfulDownloads() {
      return ((InMemoryCacheClient) cacheProtocol).getNumSuccessfulDownloads();
    }

    int getNumFailedDownloads() {
      return ((InMemoryCacheClient) cacheProtocol).getNumFailedDownloads();
    }

    ImmutableSet<Digest> findMissingDigests(Iterable<Digest> digests)
        throws IOException, InterruptedException {
      return Utils.getFromFuture(cacheProtocol.findMissingDigests(digests));
    }

    @Override
    public void close() {
      cacheProtocol.close();
    }
  }
}
