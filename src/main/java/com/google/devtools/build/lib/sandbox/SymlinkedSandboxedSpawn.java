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
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Creates an execRoot for a Spawn that contains input files as symlinks to their original
 * destination.
 */
public class SymlinkedSandboxedSpawn implements SandboxedSpawn {
  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final List<String> arguments;
  private final Map<String, String> environment;
  private final Map<PathFragment, Path> inputs;
  private final Collection<PathFragment> outputs;
  private final Set<Path> writableDirs;

  public SymlinkedSandboxedSpawn(
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
    Set<Path> createdDirs = new HashSet<>();
    cleanFileSystem(inputs.keySet());
    FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, sandboxExecRoot);
    createParentDirectoriesForInputs(createdDirs, inputs.keySet());
    createInputs(inputs);
    createWritableDirectories(createdDirs, writableDirs);
    createDirectoriesForOutputs(createdDirs, outputs);
  }

  private void cleanFileSystem(Set<PathFragment> allowedFiles) throws IOException {
    if (sandboxExecRoot.exists(Symlinks.NOFOLLOW)) {
      deleteExceptAllowedFiles(sandboxExecRoot, allowedFiles);
    }
  }

  private void deleteExceptAllowedFiles(Path root, Set<PathFragment> allowedFiles)
      throws IOException {
    for (Path p : root.getDirectoryEntries()) {
      FileStatus stat = p.stat(Symlinks.NOFOLLOW);
      if (!stat.isDirectory()) {
        if (!allowedFiles.contains(p.relativeTo(sandboxExecRoot))) {
          p.delete();
        }
      } else {
        deleteExceptAllowedFiles(p, allowedFiles);
        if (p.readdir(Symlinks.NOFOLLOW).isEmpty()) {
          p.delete();
        }
      }
    }
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
      Preconditions.checkArgument(
          dir.startsWith(sandboxExecRoot), "Bad relative path: '%s'", inputPath);
      FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, dir);
    }
  }

  private void createInputs(Map<PathFragment, Path> inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path key = sandboxExecRoot.getRelative(entry.getKey());
      FileStatus keyStat = key.statNullable(Symlinks.NOFOLLOW);
      if (keyStat != null) {
        if (keyStat.isSymbolicLink()
            && entry.getValue() != null
            && key.readSymbolicLink().equals(entry.getValue().asFragment())) {
          continue;
        }
        key.delete();
      }
      // A null value means that we're supposed to create an empty file as the input.
      if (entry.getValue() != null) {
        key.createSymbolicLink(entry.getValue());
      } else {
        FileSystemUtils.createEmptyFile(key);
      }
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
  public void copyOutputs(Path execRoot) throws IOException {
    for (PathFragment output : outputs) {
      Path source = sandboxExecRoot.getRelative(output);
      Path target = execRoot.getRelative(output);
      if (source.isFile() || source.isSymbolicLink()) {
        // Ensure the target directory exists in the real execroot. The directories for the action
        // outputs have already been created, but the spawn outputs may be different from the
        // overall action outputs. This is the case for test actions.
        FileSystemUtils.createDirectoryAndParents(target.getParentDirectory());
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
