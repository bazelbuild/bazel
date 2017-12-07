// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SymlinkedSandboxedSpawn}. */
@RunWith(JUnit4.class)
public class SymlinkedSandboxedSpawnTest extends SandboxTestCase {
  private Path workspaceDir;
  private Path sandboxDir;
  private Path execRoot;
  private Path outputsDir;

  @Before
  public final void setupTestDirs() throws IOException {
    workspaceDir = testRoot.getRelative("workspace");
    workspaceDir.createDirectory();
    sandboxDir = testRoot.getRelative("sandbox");
    sandboxDir.createDirectory();
    execRoot = sandboxDir.getRelative("execroot");
    execRoot.createDirectory();
    outputsDir = testRoot.getRelative("outputs");
    outputsDir.createDirectory();
  }

  @Test
  public void createFileSystem() throws Exception {
    Path helloTxt = workspaceDir.getRelative("hello.txt");
    FileSystemUtils.createEmptyFile(helloTxt);

    SymlinkedSandboxedSpawn symlinkedExecRoot =
        new SymlinkedSandboxedSpawn(
            sandboxDir,
            execRoot,
            ImmutableList.of("/bin/true"),
            ImmutableMap.<String, String>of(),
            ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
            ImmutableSet.of(PathFragment.create("very/output.txt")),
            ImmutableSet.of(execRoot.getRelative("wow/writable")));

    symlinkedExecRoot.createFileSystem();

    assertThat(execRoot.getRelative("such/input.txt").isSymbolicLink()).isTrue();
    assertThat(execRoot.getRelative("such/input.txt").resolveSymbolicLinks()).isEqualTo(helloTxt);
    assertThat(execRoot.getRelative("very").isDirectory()).isTrue();
    assertThat(execRoot.getRelative("wow/writable").isDirectory()).isTrue();
  }

  @Test
  public void cleanFileSystem() throws Exception {
    Path helloTxt = workspaceDir.getRelative("hello.txt");
    FileSystemUtils.createEmptyFile(helloTxt);

    SymlinkedSandboxedSpawn symlinkedExecRoot = new SymlinkedSandboxedSpawn(
        sandboxDir,
        execRoot,
        ImmutableList.of("/bin/true"),
        ImmutableMap.<String, String>of(),
        ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
        ImmutableSet.of(PathFragment.create("very/output.txt")),
        ImmutableSet.of(execRoot.getRelative("wow/writable")));
    symlinkedExecRoot.createFileSystem();

    // Pretend to do some work inside the execRoot.
    execRoot.getRelative("tempdir").createDirectory();
    FileSystemUtils.createEmptyFile(execRoot.getRelative("very/output.txt"));
    FileSystemUtils.createEmptyFile(execRoot.getRelative("wow/writable/temp.txt"));

    // Reuse the same execRoot.
    symlinkedExecRoot.createFileSystem();

    assertThat(execRoot.getRelative("such/input.txt").exists()).isTrue();
    assertThat(execRoot.getRelative("tempdir").exists()).isFalse();
    assertThat(execRoot.getRelative("very/output.txt").exists()).isFalse();
    assertThat(execRoot.getRelative("wow/writable/temp.txt").exists()).isFalse();
  }

  @Test
  public void copyOutputs() throws Exception {
    Path outputFile = execRoot.getRelative("very/output.txt");
    Path outputLink = execRoot.getRelative("very/output.link");
    Path outputDangling = execRoot.getRelative("very/output.dangling");
    Path outputDir = execRoot.getRelative("very/output.dir");
    Path outputInUncreatedTargetDir = execRoot.getRelative("uncreated/output.txt");

    SymlinkedSandboxedSpawn symlinkedExecRoot =
        new SymlinkedSandboxedSpawn(
            sandboxDir,
            execRoot,
            ImmutableList.of("/bin/true"),
            ImmutableMap.<String, String>of(),
            ImmutableMap.<PathFragment, Path>of(),
            ImmutableSet.of(
                outputFile.relativeTo(execRoot),
                outputLink.relativeTo(execRoot),
                outputDangling.relativeTo(execRoot),
                outputDir.relativeTo(execRoot),
                outputInUncreatedTargetDir.relativeTo(execRoot)),
            ImmutableSet.<Path>of());
    symlinkedExecRoot.createFileSystem();

    FileSystemUtils.createEmptyFile(outputFile);
    outputLink.createSymbolicLink(PathFragment.create("output.txt"));
    outputDangling.createSymbolicLink(PathFragment.create("doesnotexist"));
    outputDir.createDirectory();
    FileSystemUtils.createEmptyFile(outputDir.getRelative("test.txt"));
    FileSystemUtils.createEmptyFile(outputInUncreatedTargetDir);

    outputsDir.getRelative("very").createDirectory();
    symlinkedExecRoot.copyOutputs(outputsDir);

    assertThat(outputsDir.getRelative("very/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("very/output.link").isSymbolicLink()).isTrue();
    assertThat(outputsDir.getRelative("very/output.link").resolveSymbolicLinks())
        .isEqualTo(outputsDir.getRelative("very/output.txt"));
    assertThat(outputsDir.getRelative("very/output.dangling").isSymbolicLink()).isTrue();
    try {
      outputsDir.getRelative("very/output.dangling").resolveSymbolicLinks();
      fail("expected IOException");
    } catch (IOException e) {
      // Ignored.
    }
    assertThat(outputsDir.getRelative("very/output.dir").isDirectory(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("very/output.dir/test.txt").isFile(Symlinks.NOFOLLOW))
        .isTrue();
    assertThat(outputsDir.getRelative("uncreated/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
  }
}
