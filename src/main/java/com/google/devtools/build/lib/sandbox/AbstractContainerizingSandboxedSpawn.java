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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implements the general flow of a sandboxed spawn that uses a container directory to build an
 * execution root for a spawn.
 */
public abstract class AbstractContainerizingSandboxedSpawn implements SandboxedSpawn {
  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final List<String> arguments;
  private final Map<String, String> environment;
  private final Map<PathFragment, Path> inputs;
  private final Collection<PathFragment> outputs;
  private final Set<Path> writableDirs;

  public AbstractContainerizingSandboxedSpawn(
      Path sandboxPath,
      Path sandboxExecRoot,
      List<String> arguments,
      Map<String, String> environment,
      Map<PathFragment, Path> inputs,
      Collection<PathFragment> outputs,
      Set<Path> writableDirs) {
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxExecRoot;
    this.arguments = arguments;
    this.environment = environment;
    this.inputs = inputs;
    this.outputs = outputs;
    this.writableDirs = writableDirs;
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
  public void createFileSystem() throws IOException {
    createDirectories();
    createInputs(inputs);
  }

  /**
   * No input can be a child of another input, because otherwise we might try to create a symlink
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
  private void createDirectories() throws IOException {
    LinkedHashSet<Path> dirsToCreate = new LinkedHashSet<>();

    for (PathFragment path : Iterables.concat(inputs.keySet(), outputs)) {
      Preconditions.checkArgument(!path.isAbsolute());
      Preconditions.checkArgument(!path.containsUplevelReferences());
      for (int i = 0; i < path.segmentCount(); i++) {
        dirsToCreate.add(sandboxExecRoot.getRelative(path.subFragment(0, i)));
      }
    }

    for (Path path : dirsToCreate) {
      path.createDirectory();
    }

    for (Path dir : writableDirs) {
      if (dir.startsWith(sandboxExecRoot)) {
        dir.createDirectoryAndParents();
      }
    }
  }

  protected void createInputs(Map<PathFragment, Path> inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Map.Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path key = sandboxExecRoot.getRelative(entry.getKey());
      // A null value means that we're supposed to create an empty file as the input.
      if (entry.getValue() != null) {
        copyFile(entry.getValue(), key);
      } else {
        FileSystemUtils.createEmptyFile(key);
      }
    }
  }

  protected abstract void copyFile(Path source, Path target) throws IOException;

  @Override
  public void copyOutputs(Path execRoot) throws IOException {
    SandboxedSpawn.moveOutputs(outputs, sandboxExecRoot, execRoot);
  }

  @Override
  public void delete() {
    try {
      FileSystemUtils.deleteTree(sandboxPath);
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
