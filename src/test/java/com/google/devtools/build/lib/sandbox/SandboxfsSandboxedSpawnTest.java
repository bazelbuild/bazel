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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxfsSandboxedSpawn}. */
@RunWith(JUnit4.class)
public class SandboxfsSandboxedSpawnTest {
  private Path testRoot;
  private Path workspaceDir;
  private Path outerDir;
  private SandboxfsProcess sandboxfs;

  @Before
  public final void setupTestDirs() throws IOException {
    FileSystem fileSystem = new InMemoryFileSystem();
    testRoot = fileSystem.getPath(TestUtils.tmpDir());
    testRoot.createDirectoryAndParents();

    workspaceDir = testRoot.getRelative("workspace");
    workspaceDir.createDirectory();
    outerDir = testRoot.getRelative("scratch");
    outerDir.createDirectory();

    Path mountPoint = testRoot.getRelative("sandbox");
    mountPoint.createDirectory();
    sandboxfs = new FakeSandboxfsProcess(
        mountPoint.getFileSystem(), mountPoint.asFragment());
    }

  @Test
  public void testCreateFileSystem() throws Exception {
    Path helloTxt = workspaceDir.getRelative("hello.txt");
    FileSystemUtils.createEmptyFile(helloTxt);

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            "workspace",
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
                ImmutableSet.of(),
                ImmutableMap.of()),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(PathFragment.create("wow/writable")),
            /* mapSymlinkTargets= */ false,
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null);

    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    assertThat(execRoot.getRelative("such/input.txt").isSymbolicLink()).isTrue();
    assertThat(execRoot.getRelative("such/input.txt").resolveSymbolicLinks()).isEqualTo(helloTxt);
    assertThat(execRoot.getRelative("very").isDirectory()).isTrue();
    assertThat(execRoot.getRelative("wow/writable").isDirectory()).isTrue();
  }

  @Test
  public void testExecRootContainsWorkspaceName() throws Exception {
    Path helloTxt = workspaceDir.getRelative("hello.txt");
    FileSystemUtils.createEmptyFile(helloTxt);

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            "some-workspace-name",
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(ImmutableMap.of(), ImmutableSet.of(), ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableSet.of(),
            /* mapSymlinkTargets= */ false,
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null);
    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    assertThat(execRoot.getPathString()).contains("/some-workspace-name");
  }

  @Test
  public void testDelete() throws Exception {
    Path helloTxt = workspaceDir.getRelative("hello.txt");
    FileSystemUtils.createEmptyFile(helloTxt);

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            "workspace",
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
                ImmutableSet.of(),
                ImmutableMap.of()),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(PathFragment.create("wow/writable")),
            /* mapSymlinkTargets= */ false,
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null);
    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    // Pretend to do some work inside the execRoot.
    execRoot.getRelative("tempdir").createDirectory();
    FileSystemUtils.createEmptyFile(execRoot.getRelative("very/output.txt"));
    FileSystemUtils.createEmptyFile(execRoot.getRelative("wow/writable/temp.txt"));

    spawn.delete();

    assertThat(execRoot.exists()).isFalse();
  }

  @Test
  public void testCopyOutputs() throws Exception {
    // These tests are very simple because we just rely on
    // AbstractContainerizingSandboxedSpawnTest.testMoveOutputs to properly verify all corner cases.
    PathFragment outputFile = PathFragment.create("very/output.txt");

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            "workspace",
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(ImmutableMap.of(), ImmutableSet.of(), ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(outputFile), ImmutableSet.of()),
            ImmutableSet.of(),
            /* mapSymlinkTargets= */ false,
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null);
    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    FileSystemUtils.createEmptyFile(execRoot.getRelative(outputFile));

    Path outputsDir = testRoot.getRelative("outputs");
    outputsDir.getRelative(outputFile.getParentDirectory()).createDirectoryAndParents();
    spawn.copyOutputs(outputsDir);

    assertThat(outputsDir.getRelative(outputFile).isFile(Symlinks.NOFOLLOW)).isTrue();
  }

  public void testSymlinks(boolean mapSymlinkTargets) throws Exception {
    Path input1 = workspaceDir.getRelative("dir1/input-1.txt");
    input1.getParentDirectory().createDirectory();
    FileSystemUtils.createEmptyFile(input1);

    Path input2 = workspaceDir.getRelative("dir1/input-2.txt");
    input2.getParentDirectory().createDirectory();
    FileSystemUtils.createEmptyFile(input2);

    Path linkToInput1 = workspaceDir.getRelative("dir2/link-to-input-1");
    linkToInput1.getParentDirectory().createDirectory();
    linkToInput1.createSymbolicLink(PathFragment.create("../dir1/input-1.txt"));
    assertThat(linkToInput1.readSymbolicLink().isAbsolute()).isFalse();

    Path linkToInput2 = workspaceDir.getRelative("dir2/link-to-input-2");
    linkToInput2.getParentDirectory().createDirectory();
    linkToInput2.createSymbolicLink(PathFragment.create("../dir1/input-2.txt"));
    assertThat(linkToInput2.readSymbolicLink().isAbsolute()).isFalse();

    Path linkToLink = workspaceDir.getRelative("dir2/link-to-link");
    linkToLink.getParentDirectory().createDirectory();
    linkToLink.createSymbolicLink(PathFragment.create("link-to-input-2"));
    assertThat(linkToLink.readSymbolicLink().isAbsolute()).isFalse();

    Path linkToAbsolutePath = workspaceDir.getRelative("dir2/link-to-absolute-path");
    linkToAbsolutePath.getParentDirectory().createDirectory();
    Path randomPath = workspaceDir.getRelative("/some-random-path");
    FileSystemUtils.createEmptyFile(randomPath);
    linkToAbsolutePath.createSymbolicLink(randomPath.asFragment());
    assertThat(linkToAbsolutePath.readSymbolicLink().isAbsolute()).isTrue();

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            "workspace",
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(
                ImmutableMap.of(
                    PathFragment.create("dir1/input-1.txt"), input1,
                    // input2 and linkToInput2 intentionally left unmapped to verify they are mapped
                    // as symlink targets of linktoLink.
                    PathFragment.create("such/link-1.txt"), linkToInput1,
                    PathFragment.create("such/link-to-link.txt"), linkToLink,
                    PathFragment.create("such/abs-link.txt"), linkToAbsolutePath),
                ImmutableSet.of(),
                ImmutableMap.of()),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(),
            mapSymlinkTargets,
            new SynchronousTreeDeleter(),
            /* statisticsPath= */ null);

    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    // Relative symlinks must be kept as such in the sandbox and they must resolve properly.
    assertThat(execRoot.getRelative("such/link-1.txt").readSymbolicLink())
        .isEqualTo(PathFragment.create("../dir1/input-1.txt"));
    assertThat(execRoot.getRelative("such/link-1.txt").resolveSymbolicLinks()).isEqualTo(input1);
    assertThat(execRoot.getRelative("such/link-to-link.txt").readSymbolicLink())
        .isEqualTo(PathFragment.create("link-to-input-2"));
    if (mapSymlinkTargets) {
      assertThat(execRoot.getRelative("such/link-to-link.txt").resolveSymbolicLinks())
          .isEqualTo(input2);
      assertThat(execRoot.getRelative("such/link-to-input-2").readSymbolicLink())
          .isEqualTo(PathFragment.create("../dir1/input-2.txt"));
      assertThat(execRoot.getRelative("such/link-to-input-2").resolveSymbolicLinks())
          .isEqualTo(input2);
    } else {
      assertThrows(
          "Symlink resolution worked, which means the target was mapped when not expected",
          IOException.class,
          () -> execRoot.getRelative("such/link-to-link.txt").resolveSymbolicLinks());
    }

    // Targets of symlinks must have been mapped inside the sandbox only when requested.
    assertThat(execRoot.getRelative("dir1/input-1.txt").exists()).isTrue();
    if (mapSymlinkTargets) {
      assertThat(execRoot.getRelative("dir1/input-2.txt").exists()).isTrue();
    } else {
      assertThat(execRoot.getRelative("dir1/input-2.txt").exists()).isFalse();
    }

    // Absolute symlinks must be kept as such in the sandbox no matter where they point to.
    assertThat(execRoot.getRelative("such/abs-link.txt").isSymbolicLink()).isTrue();
    assertThat(execRoot.getRelative("such/abs-link.txt").readSymbolicLinkUnchecked())
        .isEqualTo(randomPath.asFragment());
  }

  @Test
  public void testSymlinks_targetsMappedIfRequested() throws Exception {
    testSymlinks(true);
  }

  @Test
  public void testSymlinks_targetsNotMappedIfNotRequested() throws Exception {
    testSymlinks(false);
  }
}
