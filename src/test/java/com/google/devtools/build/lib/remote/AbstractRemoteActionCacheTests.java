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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

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
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.AbstractRemoteActionCache.UploadManifest;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
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
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AbstractRemoteActionCache}. */
@RunWith(JUnit4.class)
public class AbstractRemoteActionCacheTests {

  private FileSystem fs;
  private Path execRoot;
  ArtifactRoot artifactRoot;
  private final DigestUtil digestUtil = new DigestUtil(DigestHashFunction.SHA256);

  private static ListeningScheduledExecutorService retryService;

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public void setUp() throws Exception {
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot");
    execRoot.createDirectoryAndParents();
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
    try {
      um.addFiles(ImmutableList.of(link));
      fail("Expected exception");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("dangling");
      assertThat(e).hasMessageThat().contains("/execroot/link");
      assertThat(e).hasMessageThat().contains("target");
    }
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
    try {
      um.addFiles(ImmutableList.of(link));
      fail("Expected exception");
    } catch (ExecException e) {
      assertThat(e).hasMessageThat().contains("symbolic link");
      assertThat(e).hasMessageThat().contains("--remote_allow_symlink_upload");
    }
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
    try {
      um.addFiles(ImmutableList.of(dir));
      fail("Expected exception");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("dangling");
      assertThat(e).hasMessageThat().contains("/execroot/dir/link");
      assertThat(e).hasMessageThat().contains("/execroot/target");
    }
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
    try {
      um.addFiles(ImmutableList.of(dir));
      fail("Expected exception");
    } catch (ExecException e) {
      assertThat(e).hasMessageThat().contains("symbolic link");
      assertThat(e).hasMessageThat().contains("dir/link");
      assertThat(e).hasMessageThat().contains("--remote_allow_symlink_upload");
    }
  }

  @Test
  public void downloadRelativeFileSymlink() throws Exception {
    AbstractRemoteActionCache cache = newTestCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFileSymlinksBuilder().setPath("a/b/link").setTarget("../../foo");
    // Doesn't check for dangling links, hence download succeeds.
    cache.download(result.build(), execRoot, null);
    Path path = execRoot.getRelative("a/b/link");
    assertThat(path.isSymbolicLink()).isTrue();
    assertThat(path.readSymbolicLink()).isEqualTo(PathFragment.create("../../foo"));
  }

  @Test
  public void downloadRelativeDirectorySymlink() throws Exception {
    AbstractRemoteActionCache cache = newTestCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectorySymlinksBuilder().setPath("a/b/link").setTarget("foo");
    // Doesn't check for dangling links, hence download succeeds.
    cache.download(result.build(), execRoot, null);
    Path path = execRoot.getRelative("a/b/link");
    assertThat(path.isSymbolicLink()).isTrue();
    assertThat(path.readSymbolicLink()).isEqualTo(PathFragment.create("foo"));
  }

  @Test
  public void downloadRelativeSymlinkInDirectory() throws Exception {
    DefaultRemoteActionCache cache = newTestCache();
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
    cache.download(result.build(), execRoot, null);
    Path path = execRoot.getRelative("dir/link");
    assertThat(path.isSymbolicLink()).isTrue();
    assertThat(path.readSymbolicLink()).isEqualTo(PathFragment.create("../foo"));
  }

  @Test
  public void downloadAbsoluteDirectorySymlinkError() throws Exception {
    AbstractRemoteActionCache cache = newTestCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectorySymlinksBuilder().setPath("foo").setTarget("/abs/link");
    try {
      cache.download(result.build(), execRoot, null);
      fail("Expected exception");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().contains("/abs/link");
      assertThat(expected).hasMessageThat().contains("absolute path");
    }
  }

  @Test
  public void downloadAbsoluteFileSymlinkError() throws Exception {
    AbstractRemoteActionCache cache = newTestCache();
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputFileSymlinksBuilder().setPath("foo").setTarget("/abs/link");
    try {
      cache.download(result.build(), execRoot, null);
      fail("Expected exception");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().contains("/abs/link");
      assertThat(expected).hasMessageThat().contains("absolute path");
    }
  }

  @Test
  public void downloadAbsoluteSymlinkInDirectoryError() throws Exception {
    DefaultRemoteActionCache cache = newTestCache();
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("/foo")))
            .build();
    Digest treeDigest = cache.addContents(tree.toByteArray());
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputDirectoriesBuilder().setPath("dir").setTreeDigest(treeDigest);
    try {
      cache.download(result.build(), execRoot, null);
      fail("Expected exception");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().contains("dir/link");
      assertThat(expected).hasMessageThat().contains("/foo");
      assertThat(expected).hasMessageThat().contains("absolute path");
    }
  }

  @Test
  public void downloadFailureMaintainsDirectories() throws Exception {
    DefaultRemoteActionCache cache = newTestCache();
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
    try {
      cache.download(result.build(), execRoot, null);
      fail("Expected exception");
    } catch (IOException expected) {
      assertThat(cache.getNumFailedDownloads()).isEqualTo(1);
      assertThat(execRoot.getRelative("outputdir").exists()).isTrue();
      assertThat(execRoot.getRelative("outputdir/outputfile").exists()).isFalse();
      assertThat(execRoot.getRelative("otherfile").exists()).isFalse();
    }
  }

  @Test
  public void onErrorWaitForRemainingDownloadsToComplete() throws Exception {
    // If one or more downloads of output files / directories fail then the code should
    // wait for all downloads to have been completed before it tries to clean up partially
    // downloaded files.

    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");

    DefaultRemoteActionCache cache = newTestCache();
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
    try {
      cache.download(result, execRoot, new FileOutErr(stdout, stderr));
      fail("Expected IOException");
    } catch (IOException e) {
      assertThat(cache.getNumSuccessfulDownloads()).isEqualTo(2);
      assertThat(cache.getNumFailedDownloads()).isEqualTo(1);
      assertThat(cache.getDownloadQueueSize()).isEqualTo(3);
      assertThat(Throwables.getRootCause(e)).hasMessageThat().isEqualTo("download failed");
    }
  }

  @Test
  public void testDownloadMinimalFiles() throws Exception {
    // Test that injecting the metadata for a remote output file works

    // arrange
    DefaultRemoteActionCache remoteCache = newTestCache();
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file1").setDigest(d1))
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file2").setDigest(d2))
            .build();

    Artifact a1 = new Artifact(PathFragment.create("file1"), artifactRoot);
    Artifact a2 = new Artifact(PathFragment.create("file2"), artifactRoot);

    MetadataInjector injector = mock(MetadataInjector.class);

    // act
    InMemoryOutput inMemoryOutput =
        remoteCache.downloadMinimal(
            r,
            ImmutableList.of(a1, a2),
            /* inMemoryOutputPath= */ null,
            new FileOutErr(),
            execRoot,
            injector);

    // assert
    assertThat(inMemoryOutput).isNull();
    verify(injector)
        .injectRemoteFile(eq(a1), eq(toBinaryDigest(d1)), eq(d1.getSizeBytes()), anyInt());
    verify(injector)
        .injectRemoteFile(eq(a2), eq(toBinaryDigest(d2)), eq(d2.getSizeBytes()), anyInt());

    Path outputBase = artifactRoot.getRoot().asPath();
    assertThat(outputBase.readdir(Symlinks.NOFOLLOW)).isEmpty();
  }

  @Test
  public void testDownloadMinimalDirectory() throws Exception {
    // Test that injecting the metadata for a tree artifact / remote output directory works

    // arrange
    DefaultRemoteActionCache remoteCache = newTestCache();
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
            ArtifactOwner.NullArtifactOwner.INSTANCE,
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
            injector);

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
  }

  @Test
  public void testDownloadMinimalDirectoryFails() throws Exception {
    // Test that we properly fail when downloading the metadata of an output
    // directory fails

    // arrange
    DefaultRemoteActionCache remoteCache = newTestCache();
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
            ArtifactOwner.NullArtifactOwner.INSTANCE,
            SpecialArtifactType.TREE);
    MetadataInjector injector = mock(MetadataInjector.class);

    // act
    try {
      remoteCache.downloadMinimal(
          r,
          ImmutableList.of(dir),
          /* inMemoryOutputPath= */ null,
          new FileOutErr(),
          execRoot,
          injector);
      fail("IOException expected");
    } catch (IOException e) {
      assertThat(e).isEqualTo(downloadTreeException);
    }
  }

  @Test
  public void testDownloadMinimalWithStdoutStderr() throws Exception {
    // Test that downloading of non-embedded stdout and stderr works

    // arrange
    DefaultRemoteActionCache remoteCache = newTestCache();
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
            r, ImmutableList.of(), /* inMemoryOutputPath= */ null, outErr, execRoot, injector);

    // assert
    assertThat(inMemoryOutput).isNull();
    assertThat(outErr.outAsLatin1()).isEqualTo("stdout");
    assertThat(outErr.errAsLatin1()).isEqualTo("stderr");

    Path outputBase = artifactRoot.getRoot().asPath();
    assertThat(outputBase.readdir(Symlinks.NOFOLLOW)).isEmpty();
  }

  @Test
  public void testDownloadMinimalWithInMemoryOutput() throws Exception {
    // Test that downloading an in memory output works

    // arrange
    DefaultRemoteActionCache remoteCache = newTestCache();
    Digest d1 = remoteCache.addContents("content1");
    Digest d2 = remoteCache.addContents("content2");
    ActionResult r =
        ActionResult.newBuilder()
            .setExitCode(0)
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file1").setDigest(d1))
            .addOutputFiles(OutputFile.newBuilder().setPath("outputs/file2").setDigest(d2))
            .build();
    Artifact a1 = new Artifact(PathFragment.create("file1"), artifactRoot);
    Artifact a2 = new Artifact(PathFragment.create("file2"), artifactRoot);
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
            injector);

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
  }

  private DefaultRemoteActionCache newTestCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    return new DefaultRemoteActionCache(options, digestUtil);
  }

  private static class DefaultRemoteActionCache extends AbstractRemoteActionCache {

    Map<Digest, ListenableFuture<byte[]>> downloadResults = new HashMap<>();
    List<ListenableFuture<?>> blockingDownloads = new ArrayList<>();
    AtomicInteger numSuccess = new AtomicInteger();
    AtomicInteger numFailures = new AtomicInteger();

    public DefaultRemoteActionCache(RemoteOptions options, DigestUtil digestUtil) {
      super(options, digestUtil);
    }

    public Digest addContents(String txt) {
      return addContents(txt.getBytes(UTF_8));
    }

    public Digest addContents(byte[] bytes) {
      Digest digest = digestUtil.compute(bytes);
      downloadResults.put(digest, Futures.immediateFuture(bytes));
      return digest;
    }

    public Digest addContents(Message m) {
      return addContents(m.toByteArray());
    }

    public Digest addException(String txt, Exception e) {
      Digest digest = digestUtil.compute(txt.getBytes(UTF_8));
      downloadResults.put(digest, Futures.immediateFailedFuture(e));
      return digest;
    }

    Digest addException(Message m, Exception e) {
      Digest digest = digestUtil.compute(m);
      downloadResults.put(digest, Futures.immediateFailedFuture(e));
      return digest;
    }

    public int getNumSuccessfulDownloads() {
      return numSuccess.get();
    }

    public int getNumFailedDownloads() {
      return numFailures.get();
    }

    public int getDownloadQueueSize() {
      return blockingDownloads.size();
    }

    @Override
    protected <T> T getFromFuture(ListenableFuture<T> f) throws IOException, InterruptedException {
      blockingDownloads.add(f);
      return Utils.getFromFuture(f);
    }

    @Nullable
    @Override
    ActionResult getCachedActionResult(ActionKey actionKey)
        throws IOException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    void upload(
        ActionKey actionKey,
        Action action,
        Command command,
        Path execRoot,
        Collection<Path> files,
        FileOutErr outErr)
        throws ExecException, IOException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    protected ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
      SettableFuture<Void> result = SettableFuture.create();
      ListenableFuture<byte[]> downloadResult = downloadResults.get(digest);
      Futures.addCallback(
          downloadResult != null
              ? downloadResult
              : Futures.immediateFailedFuture(new CacheNotFoundException(digest, digestUtil)),
          new FutureCallback<byte[]>() {
            @Override
            public void onSuccess(byte[] bytes) {
              numSuccess.incrementAndGet();
              try {
                out.write(bytes);
                out.close();
                result.set(null);
              } catch (IOException e) {
                result.setException(e);
              }
            }

            @Override
            public void onFailure(Throwable throwable) {
              numFailures.incrementAndGet();
              result.setException(throwable);
            }
          },
          MoreExecutors.directExecutor());
      return result;
    }

    @Override
    public void close() {
      throw new UnsupportedOperationException();
    }
  }
}
