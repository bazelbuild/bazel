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
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link RemoteActionFileSystem} */
@RunWith(JUnit4.class)
public class RemoteActionFileSystemTest {

  private static final DigestHashFunction HASH_FUNCTION = DigestHashFunction.SHA256;

  @Mock private RemoteActionInputFetcher inputFetcher;
  private FileSystem fs;
  private Path execRoot;
  private ArtifactRoot outputRoot;

  @Before
  public void setUp() throws IOException {
    MockitoAnnotations.initMocks(this);
    fs = new InMemoryFileSystem(new JavaClock(), HASH_FUNCTION);
    execRoot = fs.getPath("/exec");
    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, false, false, false, "out");
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
  public void testDeleteRemoteFile() throws Exception {
    // arrange
    ActionInputMap inputs = new ActionInputMap(1);
    Artifact remoteArtifact = createRemoteArtifact("remote-file", "remote contents", inputs);
    FileSystem actionFs = newRemoteActionFileSystem(inputs);

    // act
    boolean success = actionFs.delete(actionFs.getPath(remoteArtifact.getPath().getPathString()));

    // assert
    assertThat(success).isTrue();
  }

  @Test
  public void testDeleteLocalFile() throws Exception {
    // arrange
    ActionInputMap inputs = new ActionInputMap(0);
    FileSystem actionFs = newRemoteActionFileSystem(inputs);
    Path filePath = actionFs.getPath(execRoot.getPathString()).getChild("local-file");
    FileSystemUtils.writeContent(filePath, StandardCharsets.UTF_8, "local contents");

    // act
    boolean success = actionFs.delete(actionFs.getPath(filePath.getPathString()));

    // assert
    assertThat(success).isTrue();
  }

  private FileSystem newRemoteActionFileSystem(ActionInputMap inputs) {
    return new RemoteActionFileSystem(
        fs,
        execRoot.asFragment(),
        outputRoot.getRoot().asPath().relativeTo(execRoot).getPathString(),
        inputs,
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
        new RemoteFileArtifactValue(h.asBytes(), b.length, /* locationIndex= */ 1, "action-id");
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
        a, FileArtifactValue.createFromStat(path, path.stat(Symlinks.FOLLOW), true));
    return a;
  }
}
