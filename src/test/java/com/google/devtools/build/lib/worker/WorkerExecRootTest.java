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
package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.HashMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerExecRoot}. */
@RunWith(JUnit4.class)
public class WorkerExecRootTest {
  private FileSystem fileSystem;
  private Path testRoot;
  private Path workspaceDir;
  private Path sandboxDir;
  private Path execRoot;

  @Before
  public final void setupTestDirs() throws IOException {
    fileSystem = new InMemoryFileSystem();
    testRoot = fileSystem.getPath(TestUtils.tmpDir());
    testRoot.createDirectoryAndParents();

    workspaceDir = testRoot.getRelative("workspace");
    workspaceDir.createDirectory();
    sandboxDir = testRoot.getRelative("sandbox");
    sandboxDir.createDirectory();
    execRoot = sandboxDir.getRelative("execroot");
    execRoot.createDirectory();
  }

  @Test
  public void cleanFileSystem() throws Exception {
    Path workerSh = workspaceDir.getRelative("worker.sh");
    FileSystemUtils.writeContentAsLatin1(workerSh, "#!/bin/bash");

    WorkerExecRoot workerExecRoot =
        new WorkerExecRoot(
            execRoot,
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("worker.sh"), workerSh), ImmutableMap.of()),
            SandboxOutputs.create(
                ImmutableSet.of(PathFragment.create("very/output.txt")), ImmutableSet.of()),
            ImmutableSet.of(PathFragment.create("worker.sh")));
    workerExecRoot.createFileSystem();

    // Pretend to do some work inside the execRoot.
    execRoot.getRelative("tempdir").createDirectory();
    FileSystemUtils.createEmptyFile(execRoot.getRelative("very/output.txt"));
    FileSystemUtils.createEmptyFile(execRoot.getRelative("temp.txt"));
    // Modify the worker.sh so that we're able to detect whether it gets rewritten or not.
    FileSystemUtils.writeContentAsLatin1(workerSh, "#!/bin/sh");

    // Reuse the same execRoot.
    workerExecRoot.createFileSystem();

    assertThat(execRoot.getRelative("worker.sh").exists()).isTrue();
    assertThat(
            FileSystemUtils.readContent(
                execRoot.getRelative("worker.sh"), Charset.defaultCharset()))
        .isEqualTo("#!/bin/sh");
    assertThat(execRoot.getRelative("tempdir").exists()).isFalse();
    assertThat(execRoot.getRelative("very/output.txt").exists()).isFalse();
    assertThat(execRoot.getRelative("temp.txt").exists()).isFalse();
  }

  @Test
  public void createsAndCleansInputSymlinks() throws Exception {
    // Simulate existing symlinks in the exec root to check that `WorkerExecRoot` correctly deletes
    // the unnecessary ones and updates the ones that don't point to the right target.
    FileSystemUtils.ensureSymbolicLink(execRoot.getRelative("dir/input_symlink_1"), "old_content");
    FileSystemUtils.ensureSymbolicLink(execRoot.getRelative("dir/input_symlink_2"), "unchanged");
    FileSystemUtils.ensureSymbolicLink(execRoot.getRelative("dir/input_symlink_3"), "whatever");

    WorkerExecRoot workerExecRoot =
        new WorkerExecRoot(
            execRoot,
            new SandboxInputs(
                ImmutableMap.of(),
                ImmutableMap.of(
                    PathFragment.create("dir/input_symlink_1"), PathFragment.create("new_content"),
                    PathFragment.create("dir/input_symlink_2"), PathFragment.create("unchanged"))),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableSet.of());

    // This should update the `input_symlink_{1,2,3}` according to `SandboxInputs`, i.e., update the
    // first/second (alternatively leave the second unchanged) and delete the third.
    workerExecRoot.createFileSystem();

    assertThat(execRoot.getRelative("dir/input_symlink_1").readSymbolicLink())
        .isEqualTo(PathFragment.create("new_content"));
    assertThat(execRoot.getRelative("dir/input_symlink_2").readSymbolicLink())
        .isEqualTo(PathFragment.create("unchanged"));
    assertThat(execRoot.getRelative("dir/input_symlink_3").exists()).isFalse();
  }

  @Test
  public void workspaceFilesAreNotDeleted() throws Exception {
    // We want to check that `WorkerExecRoot` deletes pre-existing symlinks if they're not in the
    // `SandboxInputs`, but it does not delete the files that the symlinks point to (i.e., the files
    // in the workspace directory).

    Path neededWorkspaceFile = workspaceDir.getRelative("needed_file");
    FileSystemUtils.writeContentAsLatin1(neededWorkspaceFile, "needed workspace content");

    Path otherWorkspaceFile = workspaceDir.getRelative("other_file");
    FileSystemUtils.writeContentAsLatin1(otherWorkspaceFile, "other workspace content");

    FileSystemUtils.ensureSymbolicLink(execRoot.getRelative("needed_file"), neededWorkspaceFile);
    FileSystemUtils.ensureSymbolicLink(execRoot.getRelative("other_file"), otherWorkspaceFile);

    WorkerExecRoot workerExecRoot =
        new WorkerExecRoot(
            execRoot,
            new SandboxInputs(
                ImmutableMap.of(PathFragment.create("needed_file"), neededWorkspaceFile),
                ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableSet.of());
    workerExecRoot.createFileSystem();

    assertThat(execRoot.getRelative("needed_file").readSymbolicLink())
        .isEqualTo(neededWorkspaceFile.asFragment());
    assertThat(execRoot.getRelative("other_file").exists()).isFalse();

    assertThat(FileSystemUtils.readContent(neededWorkspaceFile, Charset.defaultCharset()))
        .isEqualTo("needed workspace content");
    assertThat(FileSystemUtils.readContent(otherWorkspaceFile, Charset.defaultCharset()))
        .isEqualTo("other workspace content");
  }

  @Test
  public void recreatesEmptyFiles() throws Exception {
    // Simulate existing non-empty file in the exec root to check that `WorkerExecRoot` will clear
    // the contents as requested by `SandboxInputs`.
    FileSystemUtils.writeContentAsLatin1(execRoot.getRelative("some_file"), "some content");

    HashMap<PathFragment, Path> inputs = new HashMap<>();
    inputs.put(PathFragment.create("some_file"), null);
    WorkerExecRoot workerExecRoot =
        new WorkerExecRoot(
            execRoot,
            new SandboxInputs(inputs, ImmutableMap.of()),
            SandboxOutputs.create(ImmutableSet.of(), ImmutableSet.of()),
            ImmutableSet.of());

    // This is interesting, because the filepath is a key in `SandboxInputs`, but its value is
    // `null`, which means "create an empty file". So after `createFileSystem` the file should be
    // empty.
    workerExecRoot.createFileSystem();

    assertThat(
            FileSystemUtils.readContent(
                execRoot.getRelative("some_file"), Charset.defaultCharset()))
        .isEmpty();
  }
}
