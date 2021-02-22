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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
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
    fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    testRoot = fileSystem.getPath(TestUtils.tmpDir());
    testRoot.createDirectoryAndParents();

    workspaceDir = testRoot.getRelative("workspace");
    workspaceDir.createDirectory();
    sandboxDir = testRoot.getRelative("sandbox");
    sandboxDir.createDirectory();
    execRoot = sandboxDir.getRelative("execroot/__main__");
    execRoot.createDirectoryAndParents();
  }

  @Test
  public void cleanFileSystem() throws Exception {
    SandboxHelper sandboxHelper =
        new SandboxHelper(workspaceDir, execRoot)
            .addAndCreateInputFile("worker.sh", "worker.sh", "#!/bin/bash")
            .addOutput("very/output.txt")
            .addWorkerFile("worker.sh");
    Path workerSh = workspaceDir.getRelative("worker.sh");

    WorkerExecRoot workerExecRoot = new WorkerExecRoot(execRoot);
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

    // Pretend to do some work inside the execRoot.
    execRoot.getRelative("tempdir").createDirectory();
    FileSystemUtils.createEmptyFile(execRoot.getRelative("very/output.txt"));
    FileSystemUtils.createEmptyFile(execRoot.getRelative("temp.txt"));
    // Modify the worker.sh so that we're able to detect whether it gets rewritten or not.
    FileSystemUtils.writeContentAsLatin1(workerSh, "#!/bin/sh");

    // Reuse the same execRoot.
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

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
    SandboxHelper sandboxHelper =
        new SandboxHelper(workspaceDir, execRoot)
            .createSymlink("dir/input_symlink_1", "old_content")
            .createSymlink("dir/input_symlink_2", "unchanged")
            .createSymlink("dir/input_symlink_3", "whatever")
            .addSymlink("dir/input_symlink_1", "new_content")
            .addSymlink("dir/input_symlink_2", "unchanged");

    WorkerExecRoot workerExecRoot = new WorkerExecRoot(execRoot);

    // This should update the `input_symlink_{1,2,3}` according to `SandboxInputs`, i.e., update the
    // first/second (alternatively leave the second unchanged) and delete the third.
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

    assertThat(execRoot.getRelative("dir/input_symlink_1").readSymbolicLink())
        .isEqualTo(PathFragment.create("new_content"));
    assertThat(execRoot.getRelative("dir/input_symlink_2").readSymbolicLink())
        .isEqualTo(PathFragment.create("unchanged"));
    assertThat(execRoot.getRelative("dir/input_symlink_3").exists()).isFalse();
  }

  @Test
  public void createsOutputDirs() throws Exception {
    SandboxHelper sandboxHelper =
        new SandboxHelper(workspaceDir, execRoot)
            .addOutput("dir/foo/bar_kt.jar")
            .addOutput("dir/foo/bar_kt.jdeps")
            .addOutput("dir/foo/bar_kt-sources.jar")
            .addOutputDir("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_sourcegenfiles")
            .addOutputDir("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_classes")
            .addOutputDir("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_temp")
            .addOutputDir("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_generated_classes");
    WorkerExecRoot workerExecRoot = new WorkerExecRoot(execRoot);
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

    assertThat(execRoot.getRelative("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_sourcegenfiles").exists())
        .isTrue();
    assertThat(execRoot.getRelative("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_classes").exists())
        .isTrue();
    assertThat(execRoot.getRelative("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_temp").exists()).isTrue();
    assertThat(
            execRoot.getRelative("dir/foo/_kotlinc/bar_kt_jvm/bar_kt_generated_classes").exists())
        .isTrue();
  }

  @Test
  public void workspaceFilesAreNotDeleted() throws Exception {
    // We want to check that `WorkerExecRoot` deletes pre-existing symlinks if they're not in the
    // `SandboxInputs`, but it does not delete the files that the symlinks point to (i.e., the files
    // in the workspace directory).
    Path neededWorkspaceFile = workspaceDir.getRelative("needed_file");
    Path otherWorkspaceFile = workspaceDir.getRelative("other_file");

    SandboxHelper sandboxHelper =
        new SandboxHelper(workspaceDir, execRoot)
            .addInputFile("needed_file", "needed_file")
            .createWorkspaceDirFile("needed_file", "needed workspace content")
            .createWorkspaceDirFile("other_file", "other workspace content")
            .createSymlink("needed_file", neededWorkspaceFile.getPathString())
            .createSymlink("other_file", otherWorkspaceFile.getPathString());

    WorkerExecRoot workerExecRoot = new WorkerExecRoot(execRoot);
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

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
    // This is interesting, because the filepath is a key in `SandboxInputs`, but its value is
    // `null`, which means "create an empty file". So after `createFileSystem` the file should be
    // empty.
    SandboxHelper sandboxHelper =
        new SandboxHelper(workspaceDir, execRoot)
            .createExecRootFile("some_file", "some content")
            .addInputFile("some_file", null);

    WorkerExecRoot workerExecRoot = new WorkerExecRoot(sandboxHelper.execRoot);
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

    assertThat(
            FileSystemUtils.readContent(
                execRoot.getRelative("some_file"), Charset.defaultCharset()))
        .isEmpty();
  }

  @Test
  public void createsAndDeletesSiblingExternalRepoFiles() throws Exception {
    // With the sibling repository layout, external repository source files are no longer symlinked
    // under <execroot>/external/<repo name>/<path>. Instead, they become *siblings* of the main
    // workspace files in that they're placed at <execroot>/../<repo name>/<path>. Simulate this
    // layout and check if inputs are correctly created and irrelevant symlinks are deleted.

    Path fooRepoDir = workspaceDir.getRelative("external_dir/foo");
    Path input1 = fooRepoDir.getRelative("bar/input1");
    Path input2 = fooRepoDir.getRelative("input2");
    Path random = fooRepoDir.getRelative("bar/random");

    SandboxHelper sandboxHelper =
        new SandboxHelper(workspaceDir, execRoot)
            .createWorkspaceDirFile("external_dir/foo/bar/input1", "This is input1.")
            .createWorkspaceDirFile("external_dir/foo/input2", "This is input2.")
            .createWorkspaceDirFile("external_dir/foo/bar/random", "This is random.")
            .createSymlink("../foo/bar/input1", input1.getPathString())
            .createSymlink("../foo/bar/random", random.getPathString())
            .addInputFile("../foo/bar/input1", input1.getPathString())
            .addInputFile("../foo/input2", input2.getPathString());

    WorkerExecRoot workerExecRoot = new WorkerExecRoot(execRoot);
    workerExecRoot.createFileSystem(
        sandboxHelper.getWorkerFiles(),
        sandboxHelper.getSandboxInputs(),
        sandboxHelper.getSandboxOutputs());

    assertThat(execRoot.getRelative("../foo/bar/input1").readSymbolicLink())
        .isEqualTo(input1.asFragment());
    assertThat(execRoot.getRelative("../foo/input2").readSymbolicLink())
        .isEqualTo(input2.asFragment());
    assertThat(execRoot.getRelative("bar/random").exists()).isFalse();
  }

  /**
   * Helper class that sets up a sandbox in a more comprehensible way. Handles setting up
   * SandboxInputs and SandboxOutputs as well as creating related files.
   */
  private static class SandboxHelper {
    /** Map from workdir-relative input path to optional real file path. */
    private final Map<PathFragment, Path> inputs = new HashMap<>();

    private final Map<PathFragment, String> virtualInputs = new HashMap<>();
    private final Map<PathFragment, PathFragment> symlinks = new HashMap<>();
    private final Map<PathFragment, Path> workerFiles = new HashMap<>();
    private final List<PathFragment> outputFiles = new ArrayList<>();
    private final List<PathFragment> outputDirs = new ArrayList<>();

    private final Path workspaceDir;
    private final Path execRoot;

    public SandboxHelper(Path workspaceDir, Path execRoot) {
      this.workspaceDir = workspaceDir;
      this.execRoot = execRoot;
    }

    /**
     * Adds a regular input file at execRootPath under execRoot, with the real file at workspacePath
     * under workspaceDir.
     */
    public SandboxHelper addInputFile(String execRootPath, String workspacePath) {
      inputs.put(
          PathFragment.create(execRootPath),
          workspacePath != null ? workspaceDir.getRelative(workspacePath) : null);
      return this;
    }

    /**
     * Adds a regular input file at execRootPath under execRoot, with the real file at workspacePath
     * under workspaceDir. The real file gets created immediately and filled with {@code contents}.
     */
    public SandboxHelper addAndCreateInputFile(
        String execRootPath, String workspacePath, String contents) throws IOException {
      addInputFile(execRootPath, workspacePath);
      Path absPath = workspaceDir.getRelative(workspacePath);
      absPath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContentAsLatin1(absPath, contents);
      return this;
    }

    /** Adds a virtual input with some contents. */
    public SandboxHelper addAndCreateVirtualInput(String execRootPath, String contents) {
      virtualInputs.put(PathFragment.create(execRootPath), contents);
      return this;
    }

    /** Adds a symlink to the inputs. */
    public SandboxHelper addSymlink(String execRootPath, String linkTo) {
      symlinks.put(PathFragment.create(execRootPath), PathFragment.create(linkTo));
      return this;
    }

    /** Adds an output file without creating it. */
    public SandboxHelper addOutput(String execRootPath) {
      outputFiles.add(PathFragment.create(execRootPath));
      return this;
    }

    /** Adds an output directory without creating it. */
    public SandboxHelper addOutputDir(String execRootPath) {
      outputDirs.add(
          PathFragment.create(execRootPath.endsWith("/") ? execRootPath : execRootPath + "/"));
      return this;
    }

    /** Adds a worker file that is created under workspaceDir and referenced under execRoot. */
    public SandboxHelper addWorkerFile(String execRootPath) {
      Path absPath = workspaceDir.getRelative(execRootPath);
      workerFiles.put(PathFragment.create(execRootPath), absPath);
      return this;
    }

    /**
     * Adds a worker file that is created under workspaceDir and referenced under execRoot. Writes
     * the content under workspaceDir.
     */
    public SandboxHelper addAndCreateWorkerFile(String execRootPath, String contents)
        throws IOException {
      addWorkerFile(execRootPath);
      Path absPath = workspaceDir.getRelative(execRootPath);
      absPath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContentAsLatin1(absPath, contents);
      return this;
    }

    /** Creates a file with {@code contents} at {@code relPath} under execRoot. */
    public SandboxHelper createExecRootFile(String execRootPath, String contents)
        throws IOException {
      Path absPath = execRoot.getRelative(execRootPath);
      absPath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContentAsLatin1(absPath, contents);
      return this;
    }

    /** Creates a file with {@code contents} at {@code relPath} under execRoot. */
    public SandboxHelper createWorkspaceDirFile(String workspaceDirPath, String contents)
        throws IOException {
      Path absPath = workspaceDir.getRelative(workspaceDirPath);
      absPath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContentAsLatin1(absPath, contents);
      return this;
    }

    /**
     * Creates a symlink from within the execroot. The destination is just what's written into the
     * symlink and thus relative to the created symlink.
     */
    public SandboxHelper createSymlink(String execRootPath, String relativeDestination)
        throws IOException {
      Path fromPath = execRoot.getRelative(execRootPath);
      FileSystemUtils.ensureSymbolicLink(fromPath, relativeDestination);
      return this;
    }

    /**
     * Creates a symlink from within the execroot, and creates the target file with the given
     * contents. The destination is just what's written into the symlink and thus relative to the
     * created symlink.
     */
    public SandboxHelper createSymlinkWithContents(
        String execRootPath, String relativeDestination, String contents) throws IOException {
      createSymlink(execRootPath, relativeDestination);
      Path fromPath = execRoot.getRelative(execRootPath);
      Path toPath = fromPath.getRelative(relativeDestination);
      toPath.getParentDirectory().createDirectoryAndParents();
      FileSystemUtils.writeContentAsLatin1(toPath, contents);
      return this;
    }

    public SandboxInputs getSandboxInputs() {
      return new SandboxInputs(
          this.inputs,
          virtualInputs.entrySet().stream()
              .map(
                  entry ->
                      ActionsTestUtil.createVirtualActionInput(entry.getKey(), entry.getValue()))
              .collect(Collectors.toSet()),
          symlinks);
    }

    public SandboxOutputs getSandboxOutputs() {
      return SandboxOutputs.create(
          ImmutableSet.copyOf(this.outputFiles), ImmutableSet.copyOf(this.outputDirs));
    }

    public ImmutableSet<PathFragment> getWorkerFiles() {
      return ImmutableSet.copyOf(workerFiles.keySet());
    }
  }
}
