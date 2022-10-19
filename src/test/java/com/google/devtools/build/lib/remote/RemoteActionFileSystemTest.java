// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteActionFileSystem} */
@RunWith(JUnit4.class)
public final class RemoteActionFileSystemTest {

  private static final DigestHashFunction HASH_FUNCTION = DigestHashFunction.SHA256;

  private final RemoteActionInputFetcher inputFetcher = mock(RemoteActionInputFetcher.class);
  private final FileSystem fs = new InMemoryFileSystem(HASH_FUNCTION);
  private final Path execRoot = fs.getPath("/exec");
  private final ArtifactRoot outputRoot =
      ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "out");

  @Before
  public void createOutputRoot() throws IOException {
    outputRoot.getRoot().asPath().createDirectoryAndParents();
  }

  @Test
  public void testGetInputStream() throws Exception {
    // arrange
    ActionInputMap inputs = new ActionInputMap(2);
    Artifact remoteArtifact = createRemoteArtifact("remote-file", "remote contents", inputs);
    Artifact localArtifact = createLocalArtifact("local-file", "local contents", inputs);
    FileSystem actionFs = newRemoteActionFileSystem(inputs);
    doAnswer(
            invocationOnMock -> {
              FileSystemUtils.writeContent(
                  remoteArtifact.getPath(), StandardCharsets.UTF_8, "remote contents");
              return Futures.immediateFuture(null);
            })
        .when(inputFetcher)
        .downloadFile(eq(remoteArtifact.getPath()), eq(inputs.getMetadata(remoteArtifact)));

    // act
    Path remoteActionFsPath = actionFs.getPath(remoteArtifact.getPath().asFragment());
    String actualRemoteContents =
        FileSystemUtils.readContent(remoteActionFsPath, StandardCharsets.UTF_8);

    // assert
    Path localActionFsPath = actionFs.getPath(localArtifact.getPath().asFragment());
    String actualLocalContents =
        FileSystemUtils.readContent(localActionFsPath, StandardCharsets.UTF_8);
    assertThat(remoteActionFsPath.getFileSystem()).isSameInstanceAs(actionFs);
    assertThat(actualRemoteContents).isEqualTo("remote contents");
    assertThat(actualLocalContents).isEqualTo("local contents");
    verify(inputFetcher)
        .downloadFile(eq(remoteArtifact.getPath()), eq(inputs.getMetadata(remoteArtifact)));
    verifyNoMoreInteractions(inputFetcher);
  }

  @Test
  public void testCreateSymbolicLink() throws InterruptedException, IOException {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    Artifact remoteArtifact = createRemoteArtifact("remote-file", "remote contents", inputs);
    Path symlink = outputRoot.getRoot().getRelative("symlink");
    FileSystem actionFs = newRemoteActionFileSystem(inputs);
    doAnswer(
            invocationOnMock -> {
              FileSystemUtils.writeContent(
                  remoteArtifact.getPath(), StandardCharsets.UTF_8, "remote contents");
              return Futures.immediateFuture(null);
            })
        .when(inputFetcher)
        .downloadFile(eq(remoteArtifact.getPath()), eq(inputs.getMetadata(remoteArtifact)));

    // act
    Path symlinkActionFs = actionFs.getPath(symlink.getPathString());
    symlinkActionFs.createSymbolicLink(actionFs.getPath(remoteArtifact.getPath().asFragment()));

    // assert
    assertThat(symlinkActionFs.getFileSystem()).isSameInstanceAs(actionFs);
    verify(inputFetcher)
        .downloadFile(eq(remoteArtifact.getPath()), eq(inputs.getMetadata(remoteArtifact)));
    String symlinkTargetContents =
        FileSystemUtils.readContent(symlinkActionFs, StandardCharsets.UTF_8);
    assertThat(symlinkTargetContents).isEqualTo("remote contents");
    verifyNoMoreInteractions(inputFetcher);
  }

  @Test
  public void exists_fileDoesNotExist_returnsFalse() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();

    assertThat(actionFs.exists(path)).isFalse();
  }

  @Test
  public void exists_localFile_returnsTrue() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();

    writeLocalFile(actionFs, path, "local contents");

    assertThat(actionFs.exists(path)).isTrue();
  }

  @Test
  public void exists_remoteFile_returnsTrue() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();

    injectRemoteFile(actionFs, path, "remote contents");

    assertThat(actionFs.exists(path)).isTrue();
  }

  @Test
  public void exists_localAndRemoteFile_returnsTrue() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();

    writeLocalFile(actionFs, path, "local contents");
    injectRemoteFile(actionFs, path, "remote contents");

    assertThat(actionFs.exists(path)).isTrue();
  }

  @Test
  public void delete_fileDoesNotExist_returnsFalse() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();

    assertThat(actionFs.delete(path)).isFalse();
  }

  @Test
  public void delete_localFile_succeeds() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    writeLocalFile(actionFs, path, "local contents");
    assertThat(actionFs.getLocalFileSystem().exists(path)).isTrue();

    boolean success = actionFs.delete(path);

    assertThat(success).isTrue();
    assertThat(actionFs.getLocalFileSystem().exists(path)).isFalse();
  }

  @Test
  public void delete_remoteFile_succeeds() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    injectRemoteFile(actionFs, path, "remote contents");
    assertThat(actionFs.getRemoteOutputTree().exists(path)).isTrue();

    boolean success = actionFs.delete(path);

    assertThat(success).isTrue();
    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.getRemoteOutputTree().exists(path)).isFalse();
  }

  @Test
  public void delete_localAndRemoteFile_succeeds() throws Exception {
    ActionInputMap inputs = new ActionInputMap(0);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs);
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    writeLocalFile(actionFs, path, "local contents");
    injectRemoteFile(actionFs, path, "remote contents");
    assertThat(actionFs.getLocalFileSystem().exists(path)).isTrue();
    assertThat(actionFs.getRemoteOutputTree().exists(path)).isTrue();

    boolean success = actionFs.delete(path);

    assertThat(success).isTrue();
    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.getLocalFileSystem().exists(path)).isFalse();
    assertThat(actionFs.getRemoteOutputTree().exists(path)).isFalse();
  }

  @Test
  public void renameTo_fileDoesNotExist_throwError() {
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem();
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    PathFragment newPath = outputRoot.getRoot().asPath().getRelative("file-new").asFragment();

    assertThrows(FileNotFoundException.class, () -> actionFs.renameTo(path, newPath));
  }

  @Test
  public void renameTo_onlyRemoteFile_renameRemoteFile() throws Exception {
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem();
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    injectRemoteFile(actionFs, path, "remote-content");
    PathFragment newPath = outputRoot.getRoot().asPath().getRelative("file-new").asFragment();

    actionFs.renameTo(path, newPath);

    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.exists(newPath)).isTrue();
    assertThat(actionFs.getRemoteOutputTree().exists(path)).isFalse();
    assertThat(actionFs.getRemoteOutputTree().exists(newPath)).isTrue();
  }

  @Test
  public void renameTo_onlyLocalFile_renameLocalFile() throws Exception {
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem();
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    writeLocalFile(actionFs, path, "local-content");
    PathFragment newPath = outputRoot.getRoot().asPath().getRelative("file-new").asFragment();

    actionFs.renameTo(path, newPath);

    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.exists(newPath)).isTrue();
    assertThat(actionFs.getLocalFileSystem().exists(path)).isFalse();
    assertThat(actionFs.getLocalFileSystem().exists(newPath)).isTrue();
  }

  @Test
  public void renameTo_localAndRemoteFile_renameBoth() throws Exception {
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem();
    PathFragment path = outputRoot.getRoot().asPath().getRelative("file").asFragment();
    injectRemoteFile(actionFs, path, "remote-content");
    writeLocalFile(actionFs, path, "local-content");
    PathFragment newPath = outputRoot.getRoot().asPath().getRelative("file-new").asFragment();

    actionFs.renameTo(path, newPath);

    assertThat(actionFs.exists(path)).isFalse();
    assertThat(actionFs.exists(newPath)).isTrue();
    assertThat(actionFs.getRemoteOutputTree().exists(path)).isFalse();
    assertThat(actionFs.getRemoteOutputTree().exists(newPath)).isTrue();
    assertThat(actionFs.getLocalFileSystem().exists(path)).isFalse();
    assertThat(actionFs.getLocalFileSystem().exists(newPath)).isTrue();
  }

  @Test
  public void createDirectoryAndParents_createLocallyAndRemotely() throws Exception {
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem();
    PathFragment path = outputRoot.getRoot().asPath().getRelative("dir").asFragment();

    actionFs.createDirectoryAndParents(path);

    assertThat(actionFs.getRemoteOutputTree().getPath(path).isDirectory()).isTrue();
    assertThat(actionFs.getLocalFileSystem().getPath(path).isDirectory()).isTrue();
  }

  @Test
  public void createDirectory_createLocallyAndRemotely() throws Exception {
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem();
    actionFs.createDirectoryAndParents(outputRoot.getRoot().asPath().asFragment());
    PathFragment path = outputRoot.getRoot().asPath().getRelative("dir").asFragment();

    actionFs.createDirectory(path);

    assertThat(actionFs.getRemoteOutputTree().getPath(path).isDirectory()).isTrue();
    assertThat(actionFs.getLocalFileSystem().getPath(path).isDirectory()).isTrue();
  }

  private void injectRemoteFile(RemoteActionFileSystem actionFs, PathFragment path, String content)
      throws IOException {
    byte[] contentBytes = content.getBytes(StandardCharsets.UTF_8);
    HashCode hashCode = HASH_FUNCTION.getHashFunction().hashBytes(contentBytes);
    actionFs.injectRemoteFile(path, hashCode.asBytes(), contentBytes.length, "action-id");
  }

  private void writeLocalFile(RemoteActionFileSystem actionFs, PathFragment path, String content)
      throws IOException {
    FileSystemUtils.writeContent(actionFs.getPath(path), StandardCharsets.UTF_8, content);
  }

  private RemoteActionFileSystem newRemoteActionFileSystem() {
    ActionInputMap inputs = new ActionInputMap(0);
    return newRemoteActionFileSystem(inputs);
  }

  private RemoteActionFileSystem newRemoteActionFileSystem(ActionInputMap inputs) {
    return newRemoteActionFileSystem(inputs, ImmutableList.of());
  }

  private RemoteActionFileSystem newRemoteActionFileSystem(
      ActionInputMap inputs, Iterable<Artifact> outputs) {
    return new RemoteActionFileSystem(
        fs,
        execRoot.asFragment(),
        outputRoot.getRoot().asPath().relativeTo(execRoot).getPathString(),
        inputs,
        outputs,
        inputFetcher);
  }

  /** Returns a remote artifact and puts its metadata into the action input map. */
  private Artifact createRemoteArtifact(
      String pathFragment, String contents, ActionInputMap inputs) {
    Path p = outputRoot.getRoot().asPath().getRelative(pathFragment);
    Artifact a = ActionsTestUtil.createArtifact(outputRoot, p);
    byte[] b = contents.getBytes(StandardCharsets.UTF_8);
    HashCode h = HASH_FUNCTION.getHashFunction().hashBytes(b);
    FileArtifactValue f =
        RemoteFileArtifactValue.create(h.asBytes(), b.length, /* locationIndex= */ 1, "action-id");
    inputs.putWithNoDepOwner(a, f);
    return a;
  }

  /** Returns a local artifact and puts its metadata into the action input map. */
  private Artifact createLocalArtifact(String pathFragment, String contents, ActionInputMap inputs)
      throws IOException {
    Path p = outputRoot.getRoot().asPath().getRelative(pathFragment);
    FileSystemUtils.writeContent(p, StandardCharsets.UTF_8, contents);
    Artifact a = ActionsTestUtil.createArtifact(outputRoot, p);
    Path path = a.getPath();
    // Caution: there's a race condition between stating the file and computing the
    // digest. We need to stat first, since we're using the stat to detect changes.
    // We follow symlinks here to be consistent with getDigest.
    inputs.putWithNoDepOwner(
        a,
        FileArtifactValue.createFromStat(path, path.stat(Symlinks.FOLLOW), SyscallCache.NO_CACHE));
    return a;
  }
}
