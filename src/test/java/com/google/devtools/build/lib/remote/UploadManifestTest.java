// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UploadManifest}. */
@RunWith(JUnit4.class)
public class UploadManifestTest {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  private Path execRoot;
  private RemotePathResolver remotePathResolver;

  @Before
  public final void setUp() throws Exception {
    FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot");
    execRoot.createDirectoryAndParents();

    remotePathResolver = new RemotePathResolver.DefaultRemotePathResolver(execRoot);
  }

  @Test
  public void actionResult_noFollowSymlinks_absoluteFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("link").setDigest(digest).setIsExecutable(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_noFollowSymlinks_absoluteDirectorySymlinkAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path foo = execRoot.getRelative("dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("link");
    link.createSymbolicLink(dir);

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, execRoot.getRelative("link/foo"));

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
  public void actionResult_followSymlinks_relativeFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("link").setDigest(digest).setIsExecutable(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_followSymlinks_relativeDirectorySymlinkAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path foo = execRoot.getRelative("dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("link");
    link.createSymbolicLink(dir.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, execRoot.getRelative("link/foo"));

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
  public void actionResult_noFollowSymlinks_relativeFileSymlinkAsSymlink() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFileSymlinksBuilder().setPath("link").setTarget("target");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_noFollowSymlinks_relativeDirectorySymlinkAsSymlink() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path file = execRoot.getRelative("dir/foo");
    FileSystemUtils.writeContent(file, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("link");
    link.createSymbolicLink(dir.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectorySymlinksBuilder().setPath("link").setTarget("dir");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_noFollowSymlinks_danglingSymlinkError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/link");
    assertThat(e).hasMessageThat().contains("target");
  }

  @Test
  public void actionResult_noFollowSymlinks_absoluteFileSymlinkInDirectoryAsFile()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
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
  public void actionResult_noFollowSymlinks_absoluteDirectorySymlinkInDirectoryAsDirectory()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path bardir = execRoot.getRelative("bardir");
    bardir.createDirectory();
    Path foo = execRoot.getRelative("bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(bardir);

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, execRoot.getRelative("dir/link/foo"));

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
  public void actionResult_followSymlinks_relativeFileSymlinkInDirectoryAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(PathFragment.create("../target"));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ true);
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
  public void actionResult_followSymlinks_relativeDirectorySymlinkInDirectoryAsDirectory()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path bardir = execRoot.getRelative("bardir");
    bardir.createDirectory();
    Path foo = execRoot.getRelative("bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, execRoot.getRelative("dir/link/foo"));

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
  public void actionResult_noFollowSymlinks_relativeFileSymlinkInDirectoryAsSymlink()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(PathFragment.create("../target"));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
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
  public void actionResult_noFollowSymlinks_relativeDirectorySymlinkInDirectoryAsSymlink()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path bardir = execRoot.getRelative("bardir");
    bardir.createDirectory();
    Path foo = execRoot.getRelative("bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
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
  public void actionResult_noFollowSymlinks_danglingSymlinkInDirectoryError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(digestUtil, remotePathResolver, result, /*followSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/dir/link");
    assertThat(e).hasMessageThat().contains("/execroot/target");
  }
}
