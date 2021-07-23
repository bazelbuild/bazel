// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Implements the general flow of a sandboxed spawn that uses a container directory to build an
 * execution root for a spawn.
 */
public abstract class AbstractContainerizingSandboxedSpawn implements SandboxedSpawn {

  final Path sandboxPath;
  final Path sandboxExecRoot;
  private final List<String> arguments;
  private final Map<String, String> environment;
  final SandboxInputs inputs;
  final SandboxOutputs outputs;
  private final Set<Path> writableDirs;
  private final TreeDeleter treeDeleter;
  private final Path statisticsPath;

  public AbstractContainerizingSandboxedSpawn(
      Path sandboxPath,
      Path sandboxExecRoot,
      List<String> arguments,
      Map<String, String> environment,
      SandboxInputs inputs,
      SandboxOutputs outputs,
      Set<Path> writableDirs,
      TreeDeleter treeDeleter,
      @Nullable Path statisticsPath) {
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxExecRoot;
    this.arguments = arguments;
    this.environment = environment;
    this.inputs = inputs;
    this.outputs = outputs;
    this.writableDirs = writableDirs;
    this.treeDeleter = treeDeleter;
    this.statisticsPath = statisticsPath;
  }

  @Override
  public Path getSandboxExecRoot() {
    return sandboxExecRoot;
  }

  @Override
  public List<String> getArguments() {
    return arguments;
  }

  @Override
  public Map<String, String> getEnvironment() {
    return environment;
  }

  @Override
  @Nullable
  public Path getStatisticsPath() {
    return statisticsPath;
  }

  @Override
  public void createFileSystem() throws IOException {
    // First compute all the inputs and directories that we need. This is based only on
    // `workerFiles`, `inputs` and `outputs` and won't do any I/O.
    Set<PathFragment> inputsToCreate = new LinkedHashSet<>();
    LinkedHashSet<PathFragment> dirsToCreate = new LinkedHashSet<>();
    Set<PathFragment> writableSandboxDirs =
        writableDirs.stream()
            .filter(p -> p.startsWith(sandboxExecRoot))
            .map(p -> p.relativeTo(sandboxExecRoot))
            .collect(Collectors.toSet());
    SandboxHelpers.populateInputsAndDirsToCreate(
        inputs, ImmutableSet.of(), outputs, writableSandboxDirs, inputsToCreate, dirsToCreate);

    // Allow subclasses to filter out inputs and dirs that don't need to be created.
    filterInputsAndDirsToCreate(inputsToCreate, dirsToCreate);

    // Finally create what needs creating.
    createDirectories(dirsToCreate);
    createInputs(inputsToCreate, inputs);
    inputs.materializeVirtualInputs(sandboxExecRoot);
  }

  protected void filterInputsAndDirsToCreate(
      Set<PathFragment> inputsToCreate, LinkedHashSet<PathFragment> dirsToCreate)
      throws IOException {}

  /**
   * Creates all directories needed for the sandbox.
   *
   * <p>No input can be a child of another input, because otherwise we might try to create a symlink
   * below another symlink we created earlier - which means we'd actually end up writing somewhere
   * in the workspace.
   *
   * <p>If all inputs were regular files, this situation could naturally not happen - but
   * unfortunately, we might get the occasional action that has directories in its inputs.
   *
   * <p>Creating all parent directories first ensures that we can safely create symlinks to
   * directories, too, because we'll get an IOException with EEXIST if inputs happen to be nested
   * once we start creating the symlinks for all inputs.
   */
  void createDirectories(Iterable<PathFragment> dirsToCreate) throws IOException {
    Set<Path> knownDirectories = new HashSet<>();
    // Add sandboxExecRoot and it's parent -- all paths must fall under the parent of
    // sandboxExecRoot and we know that sandboxExecRoot exists. This stops the recursion in
    // createDirectoryAndParentsInSandboxRoot.
    knownDirectories.add(sandboxExecRoot);
    knownDirectories.add(sandboxExecRoot.getParentDirectory());

    for (PathFragment path : dirsToCreate) {
      Preconditions.checkArgument(!path.isAbsolute(), path);
      if (path.containsUplevelReferences() && path.isMultiSegment()) {
        // Allow a single up-level reference to allow inputs from the siblings of the main
        // repository in the sandbox execution root, but forbid multiple up-level references.
        // PathFragment is normalized, so up-level references are guaranteed to be at the beginning.
        Preconditions.checkArgument(
            !PathFragment.containsUplevelReferences(path.getSegment(1)),
            "%s escapes the sandbox exec root.",
            path);
      }

      SandboxHelpers.createDirectoryAndParentsInSandboxRoot(
          sandboxExecRoot.getRelative(path), knownDirectories, sandboxExecRoot);
    }
  }

  /**
   * Creates all inputs needed for this spawn's sandbox.
   *
   * @param inputsToCreate The inputs that actually need to be created. Some inputs may already
   *     exist if we're reusing a previously existing sandbox.
   * @param inputs All the inputs for this spawn.
   */
  void createInputs(Iterable<PathFragment> inputsToCreate, SandboxInputs inputs)
      throws IOException {
    for (PathFragment fragment : inputsToCreate) {
      Path key = sandboxExecRoot.getRelative(fragment);
      if (inputs.getFiles().containsKey(fragment)) {
        Path fileDest = inputs.getFiles().get(fragment);
        if (fileDest != null) {
          copyFile(fileDest, key);
        } else {
          FileSystemUtils.createEmptyFile(key);
        }
      } else if (inputs.getSymlinks().containsKey(fragment)) {
        PathFragment symlinkDest = inputs.getSymlinks().get(fragment);
        if (symlinkDest != null) {
          key.createSymbolicLink(symlinkDest);
        }
      }
    }
  }

  protected abstract void copyFile(Path source, Path target) throws IOException;

  @Override
  public void copyOutputs(Path execRoot) throws IOException {
    SandboxHelpers.moveOutputs(outputs, sandboxExecRoot, execRoot);
  }

  @Override
  public void delete() {
    try {
      treeDeleter.deleteTree(sandboxPath);
    } catch (IOException e) {
      // This usually means that the Spawn itself exited, but still has children running that
      // we couldn't wait for, which now block deletion of the sandbox directory. On Linux this
      // should never happen, as we use PID namespaces and where they are not available the
      // subreaper feature to make sure all children have been reliably killed before returning,
      // but on other OS this might not always work. The SandboxModule will try to delete them
      // again when the build is all done, at which point it hopefully works, so let's just go
      // on here.
    }
  }
}
