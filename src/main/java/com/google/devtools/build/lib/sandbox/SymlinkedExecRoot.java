// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.io.Files;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Creates an execRoot for a Spawn that contains input files as symlinks to their original
 * destination.
 */
final class SymlinkedExecRoot implements SandboxExecRoot {

  private final Path sandboxExecRoot;

  public SymlinkedExecRoot(Path sandboxExecRoot) {
    this.sandboxExecRoot = sandboxExecRoot;
  }

  @Override
  public void createFileSystem(
      Map<PathFragment, Path> inputs, Collection<PathFragment> outputs, Set<Path> writableDirs)
      throws IOException {
    Set<Path> createdDirs = new HashSet<>();
    FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, sandboxExecRoot);
    createParentDirectoriesForInputs(createdDirs, inputs.keySet());
    createSymlinksForInputs(inputs);
    createWritableDirectories(createdDirs, writableDirs);
    createDirectoriesForOutputs(createdDirs, outputs);
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
  private void createParentDirectoriesForInputs(Set<Path> createdDirs, Set<PathFragment> inputs)
      throws IOException {
    for (PathFragment inputPath : inputs) {
      Path dir = sandboxExecRoot.getRelative(inputPath).getParentDirectory();
      Preconditions.checkArgument(dir.startsWith(sandboxExecRoot));
      FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, dir);
    }
  }

  private void createSymlinksForInputs(Map<PathFragment, Path> inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path key = sandboxExecRoot.getRelative(entry.getKey());
      key.createSymbolicLink(entry.getValue());
    }
  }

  private void createWritableDirectories(Set<Path> createdDirs, Set<Path> writableDirs)
      throws IOException {
    for (Path writablePath : writableDirs) {
      if (writablePath.startsWith(sandboxExecRoot)) {
        FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, writablePath);
      }
    }
  }

  /** Prepare the output directories in the sandbox. */
  private void createDirectoriesForOutputs(Set<Path> createdDirs, Collection<PathFragment> outputs)
      throws IOException {
    for (PathFragment output : outputs) {
      FileSystemUtils.createDirectoryAndParentsWithCache(
          createdDirs, sandboxExecRoot.getRelative(output.getParentDirectory()));
    }
  }

  /** Moves all {@code outputs} to {@code execRoot}. */
  @Override
  public void copyOutputs(Path execRoot, Collection<PathFragment> outputs) throws IOException {
    for (PathFragment output : outputs) {
      Path source = sandboxExecRoot.getRelative(output);
      Path target = execRoot.getRelative(output);
      if (source.isFile() || source.isSymbolicLink()) {
        Files.move(source.getPathFile(), target.getPathFile());
      } else if (source.isDirectory()) {
        try {
          source.renameTo(target);
        } catch (IOException e) {
          // Failed to move directory directly, thus move it recursively.
          target.createDirectory();
          FileSystemUtils.moveTreesBelow(source, target);
        }
      }
    }
  }
}
