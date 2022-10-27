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
import static com.google.common.truth.Truth8.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
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
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Tests for {@link RemoteActionFileSystem} */
@RunWith(JUnit4.class)
public final class RemoteActionFileSystemTest {

  private static final DigestHashFunction HASH_FUNCTION = DigestHashFunction.SHA256;

  private final RemoteActionInputFetcher inputFetcher = mock(RemoteActionInputFetcher.class);
  private final MetadataInjector metadataInjector = mock(MetadataInjector.class);
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
  public void createSymbolicLink_localFileArtifact() throws IOException {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    Artifact localArtifact = createLocalArtifact("local-file", "local contents", inputs);
    Artifact outputArtifact = ActionsTestUtil.createArtifact(outputRoot, "out");
    ImmutableList<Artifact> outputs = ImmutableList.of(outputArtifact);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs, outputs);

    // act
    PathFragment linkPath = outputArtifact.getPath().asFragment();
    PathFragment targetPath = localArtifact.getPath().asFragment();
    Path symlinkActionFs = actionFs.getPath(linkPath);
    symlinkActionFs.createSymbolicLink(actionFs.getPath(targetPath));

    // assert
    assertThat(symlinkActionFs.getFileSystem()).isSameInstanceAs(actionFs);
    assertThat(symlinkActionFs.readSymbolicLink()).isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);

    // act
    actionFs.flush();

    // assert
    verifyNoInteractions(metadataInjector);
  }

  @Test
  public void createSymbolicLink_remoteFileArtifact() throws IOException {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    Artifact remoteArtifact = createRemoteArtifact("remote-file", "remote contents", inputs);
    Artifact outputArtifact = ActionsTestUtil.createArtifact(outputRoot, "out");
    ImmutableList<Artifact> outputs = ImmutableList.of(outputArtifact);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs, outputs);

    // act
    PathFragment linkPath = outputArtifact.getPath().asFragment();
    PathFragment targetPath = remoteArtifact.getPath().asFragment();
    Path symlinkActionFs = actionFs.getPath(linkPath);
    symlinkActionFs.createSymbolicLink(actionFs.getPath(targetPath));

    // assert
    assertThat(symlinkActionFs.getFileSystem()).isSameInstanceAs(actionFs);
    assertThat(symlinkActionFs.readSymbolicLink()).isEqualTo(targetPath);
    assertThat(outputArtifact.getPath().readSymbolicLink()).isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);

    // act
    actionFs.flush();

    // assert
    ArgumentCaptor<FileArtifactValue> metadataCaptor =
        ArgumentCaptor.forClass(FileArtifactValue.class);
    verify(metadataInjector).injectFile(eq(outputArtifact), metadataCaptor.capture());
    assertThat(metadataCaptor.getValue()).isInstanceOf(RemoteFileArtifactValue.class);
    assertThat(metadataCaptor.getValue().getMaterializationExecPath())
        .hasValue(targetPath.relativeTo(execRoot.asFragment()));
    verifyNoMoreInteractions(metadataInjector);
  }

  @Test
  public void createSymbolicLink_localTreeArtifact() throws IOException {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    ImmutableMap<String, String> contentMap =
        ImmutableMap.of("foo", "foo contents", "bar", "bar contents");
    Artifact localArtifact = createLocalTreeArtifact("remote-dir", contentMap, inputs);
    SpecialArtifact outputArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, "out");
    ImmutableList<Artifact> outputs = ImmutableList.of(outputArtifact);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs, outputs);

    // act
    PathFragment linkPath = outputArtifact.getPath().asFragment();
    PathFragment targetPath = localArtifact.getPath().asFragment();
    Path symlinkActionFs = actionFs.getPath(linkPath);
    symlinkActionFs.createSymbolicLink(actionFs.getPath(targetPath));

    // assert
    assertThat(symlinkActionFs.getFileSystem()).isSameInstanceAs(actionFs);
    assertThat(symlinkActionFs.readSymbolicLink()).isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);

    // act
    actionFs.flush();

    // assert
    verifyNoInteractions(metadataInjector);
  }

  @Test
  public void createSymbolicLink_remoteTreeArtifact() throws IOException {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    ImmutableMap<String, String> contentMap =
        ImmutableMap.of("foo", "foo contents", "bar", "bar contents");
    Artifact remoteArtifact = createRemoteTreeArtifact("remote-dir", contentMap, inputs);
    SpecialArtifact outputArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, "out");
    ImmutableList<Artifact> outputs = ImmutableList.of(outputArtifact);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs, outputs);

    // act
    PathFragment linkPath = outputArtifact.getPath().asFragment();
    PathFragment targetPath = remoteArtifact.getPath().asFragment();
    Path symlinkActionFs = actionFs.getPath(linkPath);
    symlinkActionFs.createSymbolicLink(actionFs.getPath(targetPath));

    // assert
    assertThat(symlinkActionFs.getFileSystem()).isSameInstanceAs(actionFs);
    assertThat(symlinkActionFs.readSymbolicLink()).isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);

    // act
    actionFs.flush();

    // assert
    ArgumentCaptor<TreeArtifactValue> metadataCaptor =
        ArgumentCaptor.forClass(TreeArtifactValue.class);
    verify(metadataInjector).injectTree(eq(outputArtifact), metadataCaptor.capture());
    assertThat(metadataCaptor.getValue().getMaterializationExecPath())
        .hasValue(targetPath.relativeTo(execRoot.asFragment()));
    verifyNoMoreInteractions(metadataInjector);
  }

  @Test
  public void createSymbolicLink_unresolvedSymlink() throws IOException {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    SpecialArtifact outputArtifact =
        ActionsTestUtil.createUnresolvedSymlinkArtifact(outputRoot, "out");
    ImmutableList<Artifact> outputs = ImmutableList.of(outputArtifact);
    RemoteActionFileSystem actionFs = newRemoteActionFileSystem(inputs, outputs);
    PathFragment targetPath = PathFragment.create("some/path");

    // act
    PathFragment linkPath = outputArtifact.getPath().asFragment();
    Path symlinkActionFs = actionFs.getPath(linkPath);
    symlinkActionFs.createSymbolicLink(targetPath);

    // assert
    assertThat(symlinkActionFs.getFileSystem()).isSameInstanceAs(actionFs);
    assertThat(symlinkActionFs.readSymbolicLink()).isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);
    assertThat(actionFs.getLocalFileSystem().getPath(linkPath).readSymbolicLink())
        .isEqualTo(targetPath);

    // act
    actionFs.flush();

    // assert
    verifyNoInteractions(metadataInjector);
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
  public void renameTo_fileDoesNotExist_throwError() throws Exception {
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

  private RemoteActionFileSystem newRemoteActionFileSystem() throws IOException {
    ActionInputMap inputs = new ActionInputMap(0);
    return newRemoteActionFileSystem(inputs, ImmutableList.of());
  }

  private RemoteActionFileSystem newRemoteActionFileSystem(ActionInputMap inputs)
      throws IOException {
    return newRemoteActionFileSystem(inputs, ImmutableList.of());
  }

  private RemoteActionFileSystem newRemoteActionFileSystem(
      ActionInputMap inputs, Iterable<Artifact> outputs) throws IOException {
    RemoteActionFileSystem remoteActionFileSystem =
        new RemoteActionFileSystem(
            fs,
            execRoot.asFragment(),
            outputRoot.getRoot().asPath().relativeTo(execRoot).getPathString(),
            inputs,
            outputs,
            inputFetcher);
    remoteActionFileSystem.updateContext(metadataInjector);
    remoteActionFileSystem.createDirectoryAndParents(outputRoot.getRoot().asPath().asFragment());
    return remoteActionFileSystem;
  }

  /** Returns a remote artifact and puts its metadata into the action input map. */
  private Artifact createRemoteArtifact(
      String pathFragment, String contents, ActionInputMap inputs) {
    Artifact a = ActionsTestUtil.createArtifact(outputRoot, pathFragment);
    byte[] b = contents.getBytes(StandardCharsets.UTF_8);
    HashCode h = HASH_FUNCTION.getHashFunction().hashBytes(b);
    RemoteFileArtifactValue f =
        RemoteFileArtifactValue.create(h.asBytes(), b.length, /* locationIndex= */ 1, "action-id");
    inputs.putWithNoDepOwner(a, f);
    return a;
  }

  /** Returns a remote tree artifact and puts its metadata into the action input map. */
  private SpecialArtifact createRemoteTreeArtifact(
      String pathFragment, Map<String, String> contentMap, ActionInputMap inputs) {
    SpecialArtifact a =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, pathFragment);
    inputs.putTreeArtifact(a, createRemoteTreeArtifactValue(a, contentMap), /* depOwner= */ null);
    return a;
  }

  private TreeArtifactValue createRemoteTreeArtifactValue(
      SpecialArtifact a, Map<String, String> contentMap) {
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder(a);
    for (Map.Entry<String, String> entry : contentMap.entrySet()) {
      TreeFileArtifact child = TreeFileArtifact.createTreeOutput(a, entry.getKey());
      byte[] b = entry.getValue().getBytes(StandardCharsets.UTF_8);
      HashCode h = HASH_FUNCTION.getHashFunction().hashBytes(b);
      RemoteFileArtifactValue childMeta =
          RemoteFileArtifactValue.create(
              h.asBytes(), b.length, /* locationIndex= */ 0, "action-id");
      builder.putChild(child, childMeta);
    }
    return builder.build();
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

  /** Returns a local tree artifact and puts its metadata into the action input map. */
  private SpecialArtifact createLocalTreeArtifact(
      String pathFragment, Map<String, String> contentMap, ActionInputMap inputs)
      throws IOException {
    Path dir = outputRoot.getRoot().asPath().getRelative(pathFragment);
    dir.createDirectoryAndParents();
    for (Map.Entry<String, String> entry : contentMap.entrySet()) {
      Path child = dir.getRelative(entry.getKey());
      child.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContent(child, entry.getValue().getBytes(StandardCharsets.UTF_8));
    }
    SpecialArtifact a =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(outputRoot, pathFragment);
    inputs.putTreeArtifact(a, createLocalTreeArtifactValue(a, contentMap), /* depOwner= */ null);
    return a;
  }

  private TreeArtifactValue createLocalTreeArtifactValue(
      SpecialArtifact a, Map<String, String> contentMap) throws IOException {
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder(a);
    for (String name : contentMap.keySet()) {
      Path path = a.getPath().getRelative(name);
      TreeFileArtifact child = TreeFileArtifact.createTreeOutput(a, name);
      FileArtifactValue childMeta =
          FileArtifactValue.createFromStat(path, path.stat(Symlinks.FOLLOW), SyscallCache.NO_CACHE);
      builder.putChild(child, childMeta);
    }
    return builder.build();
  }
}
