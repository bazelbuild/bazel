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
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UploadManifest}. */
@RunWith(JUnit4.class)
public class UploadManifestTest {
  private static final FileStatus SPECIAL_FILE_STATUS =
      new FileStatus() {
        @Override
        public boolean isFile() {
          return true;
        }

        @Override
        public boolean isDirectory() {
          return false;
        }

        @Override
        public boolean isSymbolicLink() {
          return false;
        }

        @Override
        public boolean isSpecialFile() {
          return true;
        }

        @Override
        public long getSize() {
          return 0;
        }

        @Override
        public long getLastModifiedTime() {
          return 0;
        }

        @Override
        public long getLastChangeTime() {
          return 0;
        }

        @Override
        public long getNodeId() {
          return 0;
        }
      };

  private static final FileStatus DIR_FILE_STATUS =
      new FileStatus() {
        @Override
        public boolean isFile() {
          return false;
        }

        @Override
        public boolean isDirectory() {
          return true;
        }

        @Override
        public boolean isSymbolicLink() {
          return false;
        }

        @Override
        public boolean isSpecialFile() {
          return false;
        }

        @Override
        public long getSize() {
          return 0;
        }

        @Override
        public long getLastModifiedTime() {
          return 0;
        }

        @Override
        public long getLastChangeTime() {
          return 0;
        }

        @Override
        public long getNodeId() {
          return 0;
        }
      };

  private static final FileStatus SYMLINK_FILE_STATUS =
      new FileStatus() {
        @Override
        public boolean isFile() {
          return false;
        }

        @Override
        public boolean isDirectory() {
          return false;
        }

        @Override
        public boolean isSymbolicLink() {
          return true;
        }

        @Override
        public boolean isSpecialFile() {
          return false;
        }

        @Override
        public long getSize() {
          return 0;
        }

        @Override
        public long getLastModifiedTime() {
          return 0;
        }

        @Override
        public long getLastChangeTime() {
          return 0;
        }

        @Override
        public long getNodeId() {
          return 0;
        }
      };

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
  public void actionResult_followSymlinks_absoluteFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(target);
    assertThat(um.getDigestToFile()).containsExactly(digest, link);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("link").setDigest(digest).setIsExecutable(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_followSymlinks_absoluteDirectorySymlinkAsDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path foo = execRoot.getRelative("dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("link");
    link.createSymbolicLink(dir);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("link")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_noFollowSymlinks_absoluteFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("link")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("link")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFileSymlinksBuilder().setPath("link").setTarget("target");
    expectedResult.addOutputSymlinksBuilder().setPath("link").setTarget("target");
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectorySymlinksBuilder().setPath("link").setTarget("dir");
    expectedResult.addOutputSymlinksBuilder().setPath("link").setTarget("dir");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_noFollowSymlinks_noAllowDanglingSymlinks_absoluteDanglingSymlinkError()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/link");
    assertThat(e).hasMessageThat().contains("target");
  }

  @Test
  public void
      actionResult_noFollowSymlinks_allowDanglingSymlinks_noAllowAbsoluteSymlinks_absoluteDanglingSymlinkAsError()
          throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ true,
            /*allowAbsoluteSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("absolute");
    assertThat(e).hasMessageThat().contains("/execroot/link");
    assertThat(e).hasMessageThat().contains("target");
  }

  @Test
  public void
      actionResult_noFollowSymlinks_allowDanglingSymlinks_allowAbsoluteSymlinks_absoluteDanglingSymlinkAsSymlink()
          throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ true,
            /*allowAbsoluteSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFileSymlinksBuilder().setPath("link").setTarget("/execroot/target");
    expectedResult.addOutputSymlinksBuilder().setPath("link").setTarget("/execroot/target");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_noFollowSymlinks_noAllowDanglingSymlinks_relativeDanglingSymlinkError()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    link.createSymbolicLink(target.relativeTo(link.getParentDirectory()));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/link");
    assertThat(e).hasMessageThat().contains("target");
  }

  @Test
  public void actionResult_noFollowSymlinks_allowDanglingSymlinks_relativeDanglingSymlinkAsSymlink()
      throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = execRoot.getRelative("link");
    Path target = execRoot.getRelative("target");
    link.createSymbolicLink(target.relativeTo(link.getParentDirectory()));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ true,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFileSymlinksBuilder().setPath("link").setTarget("target");
    expectedResult.addOutputSymlinksBuilder().setPath("link").setTarget("target");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_followSymlinks_absoluteFileSymlinkInDirectoryAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void actionResult_followSymlinks_absoluteDirectorySymlinkInDirectoryAsDirectory()
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
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
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
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
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void
      actionResult_noFollowSymlinks_noAllowDanglingSymlinks_absoluteDanglingSymlinkInDirectory()
          throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/dir/link");
    assertThat(e).hasMessageThat().contains("/execroot/target");
  }

  @Test
  public void
      actionResult_noFollowSymlinks_allowDanglingSymlinks_absoluteDanglingSymlinkInDirectory()
          throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("target");
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ true,
            /*allowAbsoluteSymlinks=*/ false);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/dir/link");
    assertThat(e).hasMessageThat().contains("/execroot/target");
  }

  @Test
  public void
      actionResult_noFollowSymlinks_noAllowDanglingSymlinks_relativeDanglingSymlinkInDirectoryAsSymlink()
          throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("dir/target");
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target.relativeTo(link.getParentDirectory()));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(dir));
    assertThat(um.getDigestToFile()).isEmpty();

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("target")))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void
      actionResult_noFollowSymlinks_allowDanglingSymlinks_relativeDanglingSymlinkInDirectoryAsSymlink()
          throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    Path target = execRoot.getRelative("dir/target");
    Path link = execRoot.getRelative("dir/link");
    link.createSymbolicLink(target.relativeTo(link.getParentDirectory()));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ true,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(dir));
    assertThat(um.getDigestToFile()).isEmpty();

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("target")))
            .build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  // Tests to verify that files with an unsupported type (collectively, "special files") are
  // rejected. We must use mocks since Bazel's filesystems don't support their creation.

  @Test
  public void actionResult_noFollowSymlinks_specialFileError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = createDirectoryWithSpecialFile("dir", "special");
    Path special = dir.getRelative("special");

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    UserExecException e =
        assertThrows(UserExecException.class, () -> um.addFiles(ImmutableList.of(special)));
    assertThat(e).hasMessageThat().contains("special file");
    assertThat(e).hasMessageThat().contains("dir/special");
  }

  @Test
  public void actionResult_followSymlinks_specialFileSymlinkError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = createDirectoryWithSymlinkToSpecialFile("dir", "link", "special");
    Path link = dir.getRelative("link");

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    UserExecException e =
        assertThrows(UserExecException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("special file");
    assertThat(e).hasMessageThat().contains("dir/link");
  }

  @Test
  public void actionResult_noFollowSymlinks_specialFileInDirectoryError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = createDirectoryWithSpecialFile("dir", "special");

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    UserExecException e =
        assertThrows(UserExecException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("special file");
    assertThat(e).hasMessageThat().contains("dir/special");
  }

  @Test
  public void actionResult_followSymlinks_specialFileSymlinkInDirectoryError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = createDirectoryWithSymlinkToSpecialFile("dir", "link", "special");

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ true,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    UserExecException e =
        assertThrows(UserExecException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("special file");
    assertThat(e).hasMessageThat().contains("dir/link");
  }

  @Test
  public void actionResult_topologicallySortedAndDeduplicatedTree() throws Exception {
    // Create 5^3 identical files named "dir/%d/%d/%d/file".
    Path dir = execRoot.getRelative("dir");
    dir.createDirectory();
    byte[] fileContents = new byte[] {1, 2, 3, 4, 5};
    final int childrenPerDirectory = 5;
    for (int a = 0; a < childrenPerDirectory; a++) {
      Path pathA = dir.getRelative(Integer.toString(a));
      pathA.createDirectory();
      for (int b = 0; b < childrenPerDirectory; b++) {
        Path pathB = pathA.getRelative(Integer.toString(b));
        pathB.createDirectory();
        for (int c = 0; c < childrenPerDirectory; c++) {
          Path pathC = pathB.getRelative(Integer.toString(c));
          pathC.createDirectory();
          Path file = pathC.getRelative("file");
          FileSystemUtils.writeContent(file, fileContents);
        }
      }
    }

    ActionResult.Builder result = ActionResult.newBuilder();
    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*followSymlinks=*/ false,
            /*allowDanglingSymlinks=*/ false,
            /*allowAbsoluteSymlinks=*/ false);
    um.addFiles(ImmutableList.of(dir));

    // Even though we constructed 1 + 5 + 5^2 + 5^3 directories, the resulting
    // Tree message should only contain four unique instances. The directories
    // should also be topologically sorted.
    List<Directory> children = new ArrayList<>();
    Directory root =
        Directory.newBuilder()
            .addFiles(
                FileNode.newBuilder().setName("file").setDigest(digestUtil.compute(fileContents)))
            .build();
    for (int depth = 0; depth < 3; depth++) {
      Directory.Builder b = Directory.newBuilder();
      Digest parentDigest = digestUtil.compute(root.toByteArray());
      for (int i = 0; i < childrenPerDirectory; i++) {
        b.addDirectories(
            DirectoryNode.newBuilder().setName(Integer.toString(i)).setDigest(parentDigest));
      }
      children.add(0, root);
      root = b.build();
    }
    Tree tree = Tree.newBuilder().setRoot(root).addAllChildren(children).build();
    Digest treeDigest = digestUtil.compute(tree);

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult
        .addOutputDirectoriesBuilder()
        .setPath("dir")
        .setTreeDigest(treeDigest)
        .setIsTopologicallySorted(true);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  private Path createSpecialFile(String execPath) throws IOException {
    Path special = mock(Path.class);
    when(special.statIfFound(Symlinks.NOFOLLOW)).thenReturn(SPECIAL_FILE_STATUS);
    when(special.relativeTo(execRoot))
        .thenReturn(execRoot.getRelative(execPath).relativeTo(execRoot));

    return special;
  }

  private Path createSymlinkToSpecialFile(String execPath, String target) throws IOException {
    Path link = mock(Path.class);
    when(link.statIfFound(Symlinks.NOFOLLOW)).thenReturn(SYMLINK_FILE_STATUS);
    when(link.statIfFound(Symlinks.FOLLOW)).thenReturn(SPECIAL_FILE_STATUS);
    when(link.relativeTo(execRoot)).thenReturn(execRoot.getRelative(execPath).relativeTo(execRoot));
    when(link.readSymbolicLink()).thenReturn(PathFragment.create(target));

    return link;
  }

  private Path createDirectoryWithSpecialFile(String dirExecPath, String specialName)
      throws IOException {
    Path special = createSpecialFile(dirExecPath + "/" + specialName);

    Path dir = mock(Path.class);
    when(dir.statIfFound(Symlinks.NOFOLLOW)).thenReturn(DIR_FILE_STATUS);
    when(dir.readdir(Symlinks.NOFOLLOW))
        .thenReturn(ImmutableList.of(new Dirent(specialName, Dirent.Type.UNKNOWN)));
    when(dir.getRelative(specialName)).thenReturn(special);

    return dir;
  }

  private Path createDirectoryWithSymlinkToSpecialFile(
      String dirExecPath, String linkName, String specialName) throws IOException {
    Path special = createSpecialFile(dirExecPath + "/" + specialName);
    Path link = createSymlinkToSpecialFile(dirExecPath + "/" + linkName, specialName);

    Path dir = mock(Path.class);
    when(dir.statIfFound(Symlinks.NOFOLLOW)).thenReturn(DIR_FILE_STATUS);
    when(dir.readdir(Symlinks.NOFOLLOW))
        .thenReturn(
            ImmutableList.of(
                new Dirent(linkName, Dirent.Type.SYMLINK),
                new Dirent(specialName, Dirent.Type.UNKNOWN)));
    when(dir.relativeTo(execRoot))
        .thenReturn(execRoot.getRelative(dirExecPath).relativeTo(execRoot));
    when(dir.getRelative(linkName)).thenReturn(link);
    when(dir.getRelative(specialName)).thenReturn(special);

    return dir;
  }
}
