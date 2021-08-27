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
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.RemoteCache.UploadManifest;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Tests for {@link RemoteCache}. */
@RunWith(JUnit4.class)
public class RemoteCacheTests {

  private RemoteActionExecutionContext context;
  private RemotePathResolver remotePathResolver;
  private FileSystem fs;
  private Path execRoot;
  ArtifactRoot artifactRoot;
  private final DigestUtil digestUtil = new DigestUtil(DigestHashFunction.SHA256);
  private FakeActionInputFileCache fakeFileCache;

  private ListeningScheduledExecutorService retryService;

  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata("none", "none", "action-id", null);
    context = RemoteActionExecutionContext.create(metadata);
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/main");
    execRoot.createDirectoryAndParents();
    remotePathResolver = RemotePathResolver.createDefault(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "outputs");
    artifactRoot.getRoot().asPath().createDirectoryAndParents();
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @After
  public void afterEverything() throws InterruptedException {
    retryService.shutdownNow();
    retryService.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
  }

  @Test
  public void uploadAbsoluteFileSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/main/link");
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path foo = fs.getPath("/execroot/main/dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/link");
    link.createSymbolicLink(dir);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, fs.getPath("/execroot/main/link/foo"));

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
    Path link = fs.getPath("/execroot/main/link");
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ false,
            /*allowSymlinks=*/ true);
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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path foo = fs.getPath("/execroot/main/dir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/link");
    link.createSymbolicLink(dir.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ false,
            /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile()).containsExactly(digest, fs.getPath("/execroot/main/link/foo"));

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
    Path link = fs.getPath("/execroot/main/link");
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFileSymlinksBuilder().setPath("link").setTarget("target");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadRelativeDirectorySymlink() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path file = fs.getPath("/execroot/main/dir/foo");
    FileSystemUtils.writeContent(file, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/link");
    link.createSymbolicLink(dir.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).isEmpty();

    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectorySymlinksBuilder().setPath("link").setTarget("dir");
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void uploadDanglingSymlinkError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/main/link");
    Path target = fs.getPath("/execroot/main/target");
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/main/link");
    assertThat(e).hasMessageThat().contains("target");
  }

  @Test
  public void uploadSymlinksNoAllowError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/main/link");
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target.relativeTo(execRoot));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ false);
    ExecException e = assertThrows(ExecException.class, () -> um.addFiles(ImmutableList.of(link)));
    assertThat(e).hasMessageThat().contains("symbolic link");
    assertThat(e).hasMessageThat().contains("--remote_allow_symlink_upload");
  }

  @Test
  public void uploadAbsoluteFileSymlinkInDirectoryAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path bardir = fs.getPath("/execroot/main/bardir");
    bardir.createDirectory();
    Path foo = fs.getPath("/execroot/main/bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(bardir);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile())
        .containsExactly(digest, fs.getPath("/execroot/main/dir/link/foo"));

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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(PathFragment.create("../target"));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ false,
            /*allowSymlinks=*/ true);
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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path bardir = fs.getPath("/execroot/main/bardir");
    bardir.createDirectory();
    Path foo = fs.getPath("/execroot/main/bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ false,
            /*allowSymlinks=*/ true);
    um.addFiles(ImmutableList.of(dir));
    Digest digest = digestUtil.compute(foo);
    assertThat(um.getDigestToFile())
        .containsExactly(digest, fs.getPath("/execroot/main/dir/link/foo"));

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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(PathFragment.create("../target"));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
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
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path bardir = fs.getPath("/execroot/main/bardir");
    bardir.createDirectory();
    Path foo = fs.getPath("/execroot/main/bardir/foo");
    FileSystemUtils.writeContent(foo, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
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
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ true);
    IOException e = assertThrows(IOException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("dangling");
    assertThat(e).hasMessageThat().contains("/execroot/dir/link");
    assertThat(e).hasMessageThat().contains("/execroot/target");
  }

  @Test
  public void uploadSymlinkInDirectoryNoAllowError() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/main/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/main/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/main/dir/link");
    link.createSymbolicLink(target);

    UploadManifest um =
        new UploadManifest(
            digestUtil,
            remotePathResolver,
            result,
            /*uploadSymlinks=*/ true,
            /*allowSymlinks=*/ false);
    ExecException e = assertThrows(ExecException.class, () -> um.addFiles(ImmutableList.of(dir)));
    assertThat(e).hasMessageThat().contains("symbolic link");
    assertThat(e).hasMessageThat().contains("dir/link");
    assertThat(e).hasMessageThat().contains("--remote_allow_symlink_upload");
  }





  @Test
  public void testDownloadEmptyBlobAndFile() throws Exception {
    // Test that downloading an empty BLOB/file does not try to perform a download.

    // arrange
    Path file = fs.getPath("/execroot/file");
    RemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);

    // act and assert
    assertThat(Utils.getFromFuture(remoteCache.downloadBlob(context, emptyDigest))).isEmpty();

    try (OutputStream out = file.getOutputStream()) {
      Utils.getFromFuture(remoteCache.downloadFile(context, file, emptyDigest));
    }
    assertThat(file.exists()).isTrue();
    assertThat(file.getFileSize()).isEqualTo(0);
  }

  @Test
  public void downloadOutErr_empty_doNotPerformDownload() throws Exception {
    // Test that downloading empty stdout/stderr does not try to perform a download.

    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);
    ActionResult.Builder result = ActionResult.newBuilder();
    result.setStdoutDigest(emptyDigest);
    result.setStderrDigest(emptyDigest);

    RemoteCache.waitForBulkTransfer(
        remoteCache.downloadOutErr(
            context,
            result.build(),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr"))),
        true);

    assertThat(remoteCache.getNumSuccessfulDownloads()).isEqualTo(0);
    assertThat(remoteCache.getNumFailedDownloads()).isEqualTo(0);
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
    cas.put(helloDigest, "hello-contents".getBytes(StandardCharsets.UTF_8));

    Path file = fs.getPath("/execroot/symlink-to-file");
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteDownloadSymlinkTemplate = "/home/alice/cas/{hash}-{size_bytes}";
    RemoteCache remoteCache = new InMemoryRemoteCache(cas, options, digestUtil);

    // act
    Utils.getFromFuture(remoteCache.downloadFile(context, file, helloDigest));

    // assert
    assertThat(file.isSymbolicLink()).isTrue();
    assertThat(file.readSymbolicLink())
        .isEqualTo(
            PathFragment.create(
                "/home/alice/cas/a378b939ad2e1d470a9a28b34b0e256b189e85cb236766edc1d46ec3b6ca82e5-14"));
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
            context,
            remotePathResolver,
            digestUtil.asActionKey(actionDigest),
            action,
            cmd,
            ImmutableList.of(fooFile, barDir),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputFilesBuilder().setPath("a/foo").setDigest(fooDigest);
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());

    ImmutableList<Digest> toQuery =
        ImmutableList.of(fooDigest, quxDigest, barDigest, cmdDigest, actionDigest);
    assertThat(remoteCache.findMissingDigests(context, toQuery)).isEmpty();
  }

  @Test
  public void upload_emptyBlobAndFile_doNotPerformUpload() throws Exception {
    // Test that uploading an empty BLOB/file does not try to perform an upload.
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "");
    Path file = execRoot.getRelative("file");

    Utils.getFromFuture(remoteCache.uploadBlob(context, emptyDigest, ByteString.EMPTY));
    assertThat(remoteCache.findMissingDigests(context, ImmutableSet.of(emptyDigest)))
        .containsExactly(emptyDigest);

    Utils.getFromFuture(remoteCache.uploadFile(context, emptyDigest, file));
    assertThat(remoteCache.findMissingDigests(context, ImmutableSet.of(emptyDigest)))
        .containsExactly(emptyDigest);
  }

  @Test
  public void upload_emptyOutputs_doNotPerformUpload() throws Exception {
    // Test that uploading an empty output does not try to perform an upload.

    // arrange
    Digest emptyDigest =
        fakeFileCache.createScratchInput(ActionInputHelper.fromPath("bar/test/wobble"), "");
    Path file = execRoot.getRelative("bar/test/wobble");
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Action action = Action.getDefaultInstance();
    ActionKey actionDigest = digestUtil.computeActionKey(action);
    Command cmd = Command.getDefaultInstance();

    // act
    remoteCache.upload(
        context,
        remotePathResolver,
        actionDigest,
        action,
        cmd,
        ImmutableList.of(file),
        new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    assertThat(remoteCache.findMissingDigests(context, ImmutableSet.of(emptyDigest)))
        .containsExactly(emptyDigest);
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
            context,
            remotePathResolver,
            actionDigest,
            action,
            cmd,
            ImmutableList.of(barDir),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());
    assertThat(remoteCache.findMissingDigests(context, ImmutableList.of(barDigest))).isEmpty();
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
            context,
            remotePathResolver,
            actionDigest,
            action,
            cmd,
            ImmutableList.of(barDir),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")));

    // assert
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult.addOutputDirectoriesBuilder().setPath("bar").setTreeDigest(barDigest);
    assertThat(result).isEqualTo(expectedResult.build());

    ImmutableList<Digest> toQuery = ImmutableList.of(wobbleDigest, quxDigest, barDigest);
    assertThat(remoteCache.findMissingDigests(context, toQuery)).isEmpty();
  }

  private InMemoryRemoteCache newRemoteCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    return new InMemoryRemoteCache(options, digestUtil);
  }
}
