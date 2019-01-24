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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
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
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(PathFragment.create("wow/writable")));

    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    assertThat(execRoot.getRelative("such/input.txt").isSymbolicLink()).isTrue();
    assertThat(execRoot.getRelative("such/input.txt").resolveSymbolicLinks()).isEqualTo(helloTxt);
    assertThat(execRoot.getRelative("very").isDirectory()).isTrue();
    assertThat(execRoot.getRelative("wow/writable").isDirectory()).isTrue();
  }

  @Test
  public void testDelete() throws Exception {
    Path helloTxt = workspaceDir.getRelative("hello.txt");
    FileSystemUtils.createEmptyFile(helloTxt);

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(PathFragment.create("wow/writable")));
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
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            ImmutableMap.of(),
            SandboxOutputs.create(ImmutableSet.of(outputFile), ImmutableSet.of()),
            ImmutableSet.of());
    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    FileSystemUtils.createEmptyFile(execRoot.getRelative(outputFile));

    Path outputsDir = testRoot.getRelative("outputs");
    outputsDir.getRelative(outputFile.getParentDirectory()).createDirectoryAndParents();
    spawn.copyOutputs(outputsDir);

    assertThat(outputsDir.getRelative(outputFile).isFile(Symlinks.NOFOLLOW)).isTrue();
  }

  @Test
  public void testSymlinksAreKeptAsIs() throws Exception {
    Path helloTxt = workspaceDir.getRelative("dir1/hello.txt");
    helloTxt.getParentDirectory().createDirectory();
    FileSystemUtils.createEmptyFile(helloTxt);

    Path linkToHello = workspaceDir.getRelative("dir2/link-to-hello");
    linkToHello.getParentDirectory().createDirectory();
    PathFragment linkTarget = PathFragment.create("../dir1/hello.txt");
    linkToHello.createSymbolicLink(linkTarget);

    // Ensure that the symlink we have created has a relative target, as otherwise we wouldn't
    // exercise the functionality we are trying to test.
    assertThat(linkToHello.readSymbolicLink().isAbsolute()).isFalse();

    SandboxedSpawn spawn =
        new SandboxfsSandboxedSpawn(
            sandboxfs,
            outerDir,
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            ImmutableMap.of(PathFragment.create("such/input.txt"), linkToHello),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of());

    spawn.createFileSystem();
    Path execRoot = spawn.getSandboxExecRoot();

    assertThat(execRoot.getRelative("such/input.txt").isSymbolicLink()).isTrue();
    assertThat(execRoot.getRelative("such/input.txt").readSymbolicLink()).isEqualTo(linkTarget);
  }
}
