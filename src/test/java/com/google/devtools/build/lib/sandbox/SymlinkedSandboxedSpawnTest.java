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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SymlinkedSandboxedSpawn}. */
@RunWith(JUnit4.class)
public class SymlinkedSandboxedSpawnTest {
  private final ManualClock clock = new ManualClock();
  private Path workspaceDir;
  private Path sandboxDir;
  private Path execRoot;
  private Path outputsDir;

  @Before
  public final void setupTestDirs() throws IOException {
    FileSystem fileSystem = new InMemoryFileSystem(clock, DigestHashFunction.SHA256);
    Path testRoot = fileSystem.getPath(TestUtils.tmpDir());
    testRoot.createDirectoryAndParents();

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
            ImmutableMap.of(),
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("such/input.txt"), helloTxt),
                ImmutableMap.of(),
                ImmutableMap.of()),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(execRoot.getRelative("wow/writable")),
            new SynchronousTreeDeleter(),
            /* sandboxDebugPath= */ null,
            /* statisticsPath= */ null,
            /* interactiveDebugArguments= */ null,
            "SomeMnemonic",
            /* targetLabel= */ null);

    symlinkedExecRoot.createFileSystem();

    assertThat(execRoot.getRelative("such/input.txt").isSymbolicLink()).isTrue();
    assertThat(execRoot.getRelative("such/input.txt").resolveSymbolicLinks()).isEqualTo(helloTxt);
    assertThat(execRoot.getRelative("very").isDirectory()).isTrue();
    assertThat(execRoot.getRelative("wow/writable").isDirectory()).isTrue();
  }

  @Test
  public void createFileSystem_restoresInputsModifiedBeforeStashing() throws Exception {
    Path workDir = execRoot.getRelative("ws");
    Path source = workspaceDir.getRelative("input.txt");
    FileSystemUtils.writeContentAsLatin1(source, "original");
    Map<PathFragment, Path> files = new HashMap<>();
    files.put(PathFragment.create("pkg/input.txt"), source);
    files.put(PathFragment.create("pkg/empty"), null);
    SandboxOptions options =
        Options.parse(
                SandboxOptions.class,
                "--reuse_sandbox_directories",
                "--experimental_inmemory_sandbox_stashes")
            .getOptions();
    SandboxStash.initialize(
        "ws", sandboxDir.getParentDirectory(), options, new SynchronousTreeDeleter());
    try {
      SymlinkedSandboxedSpawn spawn =
          new SymlinkedSandboxedSpawn(
              sandboxDir,
              workDir,
              ImmutableList.of("/bin/true"),
              ImmutableMap.of(),
              new SandboxInputs(files, ImmutableMap.of(), ImmutableMap.of()),
              SandboxOutputs.getEmptyInstance(),
              ImmutableSet.of(),
              new SynchronousTreeDeleter(),
              /* sandboxDebugPath= */ null,
              /* statisticsPath= */ null,
              /* interactiveDebugArguments= */ null,
              "SomeMnemonic",
              /* targetLabel= */ null);
      spawn.createFileSystem();
      // The filesystem uses a deterministic clock; replace the wall-clock setup timestamp too.
      SandboxStash.setLastModified(sandboxDir, clock.advanceMillis(1000));
      clock.advanceMillis(1);
      Path link = workDir.getRelative("pkg/input.txt");
      link.delete();
      link.createSymbolicLink(PathFragment.create("/other/input.txt"));
      FileSystemUtils.appendIsoLatin1(workDir.getRelative("pkg/empty"), "modified");
      Path unexpected = workDir.getRelative("pkg/unexpected");
      FileSystemUtils.createEmptyFile(unexpected);

      spawn.delete();
      assertThat(sandboxDir.exists()).isFalse();
      assertThat(SandboxStash.getInstanceForTesting().getReadyStashesForTesting().get("SomeMnemonic"))
          .hasSize(1);
      spawn.createFileSystem();

      assertThat(link.readSymbolicLink()).isEqualTo(source.asFragment());
      assertThat(workDir.getRelative("pkg/empty").isFile(Symlinks.NOFOLLOW)).isTrue();
      assertThat(workDir.getRelative("pkg/empty").getFileSize()).isEqualTo(0);
      assertThat(unexpected.exists()).isFalse();
      assertThat(SandboxStash.getInstanceForTesting().getReadyStashesForTesting().get("SomeMnemonic"))
          .isEmpty();
    } finally {
      SandboxStash.initialize(
          "ws",
          sandboxDir.getParentDirectory(),
          Options.parse(SandboxOptions.class, "--noreuse_sandbox_directories").getOptions(),
          null);
    }
  }

  @Test
  public void copyOutputs() throws Exception {
    // These tests are very simple because we just rely on
    // AbstractContainerizingSandboxedSpawnTest.testMoveOutputs to properly verify all corner cases.
    Path outputFile = execRoot.getRelative("very/output.txt");

    SymlinkedSandboxedSpawn symlinkedExecRoot =
        new SymlinkedSandboxedSpawn(
            sandboxDir,
            execRoot,
            ImmutableList.of("/bin/true"),
            ImmutableMap.of(),
            new SandboxInputs(ImmutableMap.of(), ImmutableMap.of(), ImmutableMap.of()),
            SandboxOutputs.create(
                ImmutableSet.of(outputFile.relativeTo(execRoot)), ImmutableSet.of()),
            ImmutableSet.of(),
            new SynchronousTreeDeleter(),
            /* sandboxDebugPath= */ null,
            /* statisticsPath= */ null,
            /* interactiveDebugArguments= */ null,
            "SomeMnemonic",
            /* targetLabel= */ null);
    symlinkedExecRoot.createFileSystem();

    FileSystemUtils.createEmptyFile(outputFile);

    outputsDir.getRelative("very").createDirectory();
    symlinkedExecRoot.copyOutputs(outputsDir);

    assertThat(outputsDir.getRelative("very/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
  }
}
